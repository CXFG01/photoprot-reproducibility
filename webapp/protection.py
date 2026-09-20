"""Bounded single-worker admission control; no database or extra proxy required."""
import ipaddress
import json
import math
import time
from collections import OrderedDict

MAX_BYTES = 10 * 1024 * 1024
ALLOWED_HOSTS = {'photoprot.uk', 'api.photoprot.uk', '127.0.0.1', 'localhost'}


class Bucket:
    def __init__(self, capacity, per_minute, now):
        self.capacity, self.rate = capacity, per_minute / 60
        self.tokens, self.updated = float(capacity), now

    def take(self, now):
        self.tokens = min(self.capacity, self.tokens + max(0, now-self.updated)*self.rate)
        self.updated = now
        if self.tokens < 1:
            return max(1, math.ceil((1-self.tokens)/self.rate))
        self.tokens -= 1
        return 0


class Admission:
    """One process only. Unknown clients fail closed if the bounded map fills."""
    def __init__(self, clock=time.monotonic, max_clients=10000):
        self.clock, self.max_clients = clock, max_clients
        self.clients = OrderedDict()
        now=clock()
        self.global_search=Bucket(6,30,now)
        self.global_api=Bucket(100,600,now)

    def check(self, identity, search):
        now=self.clock()
        while self.clients and next(iter(self.clients.values()))[0] < now-600:
            self.clients.popitem(last=False)
        if identity not in self.clients:
            if len(self.clients)>=self.max_clients:return 60
            self.clients[identity]=(now,Bucket(3,10,now),Bucket(30,120,now))
        _, query, api = self.clients.pop(identity)
        self.clients[identity]=(now,query,api)
        delay=(query if search else api).take(now)
        if delay:return delay
        return (self.global_search if search else self.global_api).take(now)


def client_identity(scope, headers):
    # Uvicorn must run with --no-proxy-headers: only the actual local tunnel
    # peer may assert CF-Connecting-IP. Never use X-Forwarded-For here.
    peer=(scope.get('client') or ('unknown',0))[0]
    asserted=headers.get(b'cf-connecting-ip',b'').decode('ascii',errors='ignore')
    raw=asserted if peer in ('127.0.0.1','::1') and asserted else peer
    try:
        address=ipaddress.ip_address(raw)
        if address.version==6:
            return str(ipaddress.ip_network(str(address)+'/64',strict=False))
        return str(address)
    except ValueError:return 'unknown'


class Protection:
    def __init__(self, app, admission=None):
        self.app=app; self.admission=admission or Admission();self.active=0

    async def __call__(self, scope, receive, send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        headers=dict(scope['headers']);path=scope['path'];method=scope['method']
        async def response(status,detail,retry=None):
            out=[(b'content-type',b'application/json'),(b'cache-control',b'no-store'),(b'x-content-type-options',b'nosniff')]
            if retry:out.append((b'retry-after',str(retry).encode()))
            await send({'type':'http.response.start','status':status,'headers':out})
            await send({'type':'http.response.body','body':json.dumps({'detail':detail}).encode()})
        host=headers.get(b'host',b'').decode('ascii',errors='ignore').split(':')[0].lower()
        if host not in ALLOWED_HOSTS:return await response(400,'Unknown host.')
        if len(scope.get('query_string',b''))>2048 or len(path)>256:
            return await response(414,'Request URL is too long.')
        if method not in ('GET','HEAD','POST') or (method=='POST' and path!='/api/search'):
            return await response(405,'Method not allowed.')
        if method=='POST':
            size=headers.get(b'content-length')
            if size and (len(size)>12 or not size.isdigit() or int(size)>MAX_BYTES):
                return await response(413,'Please upload an image smaller than 10 MB.')
            origin=headers.get(b'origin')
            if origin and origin not in (b'https://photoprot.uk',b'https://api.photoprot.uk'):
                return await response(403,'Cross-site uploads are not allowed.')
            mime=headers.get(b'content-type',b'').split(b';')[0].lower()
            if mime not in (b'image/png',b'image/jpeg',b'image/webp',b'application/octet-stream'):
                return await response(415,'Send a PNG, JPEG or WebP image.')
        if path.startswith('/api/'):
            retry=self.admission.check(client_identity(scope,headers),path=='/api/search')
            if retry:return await response(429,'Request limit reached. Please wait before trying again.',retry)
        if self.active>=24:return await response(503,'Service is busy. Please try again shortly.',5)
        self.active+=1
        async def secured(message):
            if message['type']=='http.response.start':
                h=list(message.get('headers',[]))
                h.extend([(b'x-content-type-options',b'nosniff'),(b'x-frame-options',b'DENY'),
                          (b'referrer-policy',b'strict-origin-when-cross-origin'),
                          (b'permissions-policy',b'camera=(), microphone=(), geolocation=()'),
                          (b'strict-transport-security',b'max-age=31536000')])
                if path.startswith('/api/') and not any(k.lower()==b'cache-control' for k,v in h):
                    h.append((b'cache-control',b'no-store'))
                message={**message,'headers':h}
            await send(message)
        try:await self.app(scope,receive,secured)
        finally:self.active-=1
