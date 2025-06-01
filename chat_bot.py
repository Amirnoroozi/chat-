
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import threading
import queue
import uuid
import time
import asyncio
import numpy as np
import faiss
from datetime import datetime
from enum import Enum
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional, List, Tuple, Dict

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def generate_local_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

async def embed_local(text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        hidden = model.base_model(**inputs, output_hidden_states=True).hidden_states[-1]
        emb = hidden.mean(dim=1).cpu().numpy().flatten().tolist()
    return emb

class MessageCategory(str, Enum):
    IRRELEVANT = "بی ربط"
    QUESTION = "سوال"
    BUG = "باگ"
    OUTAGE = "از دسترس خارج شدن سرویس"

class RequestStatus(str, Enum):
    WAITING = "waiting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = model_name

class ChatResponse(BaseModel):
    request_id: str
    user_message: str
    ai_response: str
    model_used: str
    status: RequestStatus
    categories: List[MessageCategory]
    queue_position: Optional[int] = None
    processing_time: Optional[float] = None
    wait_time: Optional[float] = None

class SearchResult(BaseModel):
    request_id: str
    user_message: str
    ai_response: str
    distance: float

class SequentialChatQueue:
    def __init__(self, emb_dim=768):
        self.queue = queue.Queue()
        self.results = {}
        self.lock = threading.Lock()
        self.index = faiss.IndexFlatL2(emb_dim)
        self.id_map = []
        threading.Thread(target=self._worker, daemon=True).start()

    def enqueue(self, req):
        text = req.message.lower()
        cats = []
        for cat, kws in {
            MessageCategory.QUESTION: ['?', 'چه', 'چطور', 'چگونه', 'چرا', 'ایا'],
            MessageCategory.BUG: ['خطا', 'error', 'bug', 'استثنا'],
            MessageCategory.OUTAGE: ['down', 'قطع', 'از دسترس', 'دسترسی ندارم']
        }.items():
            if any(kw in text for kw in kws): cats.append(cat)
        if not cats: cats.append(MessageCategory.IRRELEVANT)
        req_id = str(uuid.uuid4())
        now = datetime.utcnow()
        self.results[req_id] = {
            'message': req.message,
            'model': req.model,
            'status': RequestStatus.WAITING,
            'categories': cats,
            'created_at': now
        }
        self.queue.put(req_id)
        return ChatResponse(
            request_id=req_id,
            user_message=req.message,
            ai_response="",
            model_used=req.model,
            status=RequestStatus.WAITING,
            categories=cats,
            queue_position=self.queue.qsize()
        )

    def _worker(self):
        while True:
            rid = self.queue.get()
            item = self.results[rid]
            with self.lock:
                item['status'] = RequestStatus.PROCESSING
                start = time.time()
                prompt = item['message']
                resp = generate_local_response(prompt)
                proc_time = time.time() - start
                wait_time = proc_time
                item.update({
                    'ai_response': resp,
                    'status': RequestStatus.COMPLETED,
                    'processing_time': proc_time,
                    'wait_time': wait_time,
                    'completed_at': datetime.utcnow()
                })
                emb = asyncio.get_event_loop().run_until_complete(embed_local(prompt + ' ' + resp))
                vec = np.array(emb, dtype='float32').reshape(1, -1)
                self.index.add(vec)
                self.id_map.append(rid)
            self.queue.task_done()

    def get_tasks(self):
        tasks = []
        for rid, d in self.results.items():
            if MessageCategory.BUG in d['categories'] or MessageCategory.OUTAGE in d['categories']:
                tasks.append({**d, 'request_id': rid})
        tasks.sort(key=lambda x: (MessageCategory.BUG not in x['categories'], x['created_at']))
        return tasks

    def get_metrics(self):
        total = len(self.results)
        times = [d['processing_time'] for d in self.results.values() if d.get('processing_time')]
        avg_ms = int(sum(times)/len(times)*1000) if times else 0
        counts = {}
        for d in self.results.values():
            for c in d['categories']:
                counts[c.value] = counts.get(c.value, 0) + 1
        return {'total_requests': total, 'average_response_time_ms': avg_ms, 'category_counts': counts}

    def search(self, q, k=10):
        emb = asyncio.get_event_loop().run_until_complete(embed_local(q))
        vec = np.array(emb, dtype='float32').reshape(1, -1)
        dists, idxs = self.index.search(vec, k)
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            if idx < len(self.id_map):
                rid = self.id_map[idx]
                d = self.results[rid]
                results.append(SearchResult(
                    request_id=rid,
                    user_message=d['message'],
                    ai_response=d.get('ai_response', ''),
                    distance=float(dist)
                ))
        return results

    def autochat(self, req):
        history = [d for d in self.results.values() if d['status'] == RequestStatus.COMPLETED]
        history = sorted(history, key=lambda x: x['completed_at'], reverse=True)[:5]
        prompt = ''
        for d in history:
            prompt += f"User: {d['message']}\nAssistant: {d['ai_response']}\n"
        prompt += f"User: {req.message}\nAssistant:"
        resp = generate_local_response(prompt)
        return self.enqueue(ChatRequest(message=req.message, model=req.model)).copy(update={'ai_response': resp, 'status': RequestStatus.COMPLETED})

app = FastAPI(title="Sequential Queue Chat API")
chat_queue = SequentialChatQueue()

@app.post('/chat', response_model=ChatResponse)
async def chat_endpoint(request):
    return chat_queue.enqueue(request)

@app.get('/tasks')
def tasks_endpoint():
    return chat_queue.get_tasks()

@app.get('/metrics')
def metrics_endpoint():
    return chat_queue.get_metrics()

@app.get('/search', response_model=List[SearchResult])
def search_endpoint(q: str = Query(..., alias='q')):
    return chat_queue.search(q)

@app.post('/autochat', response_model=ChatResponse)
async def autochat_endpoint(request):
    return chat_queue.autochat(request)

@app.get('/')
def root():
    return {'message': 'Sequential Queue Chat API'}

if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False)