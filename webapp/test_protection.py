"""Small deterministic abuse tests; never load the GPU or flood the live host."""
import asyncio
import unittest
from webapp.protection import Admission, Protection, client_identity, MAX_BYTES


class ProtectionTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, guard, path='/api/search', method='POST', extra=(), peer='127.0.0.1'):
        sent=[]
        async def receive():return {'type':'http.request','body':b'','more_body':False}
        async def send(message):sent.append(message)
        scope={'type':'http','method':method,'path':path,'query_string':b'',
               'headers':[(b'host',b'api.photoprot.uk'),(b'content-type',b'image/png'),*extra], 'client':(peer,1)}
        await guard(scope,receive,send)
        return sent[0]['status'],dict(sent[0]['headers'])

    async def ok(self,scope,receive,send):
        await send({'type':'http.response.start','status':200,'headers':[]})
        await send({'type':'http.response.body','body':b'ok'})

    async def test_limits_before_body_or_application(self):
        async def forbidden(*args):self.fail('Rejected request reached application')
        for extra,expected in [([(b'content-length',str(MAX_BYTES+1).encode())],413),
                               ([(b'origin',b'https://evil.example')],403),
                               ([(b'host',b'old.trycloudflare.com')],400),
                               ([(b'content-type',b'text/plain')],415)]:
            status,_=await self.request(Protection(forbidden),extra=extra)
            self.assertEqual(status,expected)

    async def test_search_rate_limit_and_retry(self):
        guard=Protection(self.ok,Admission(clock=lambda:0))
        for _ in range(3):self.assertEqual((await self.request(guard))[0],200)
        status,headers=await self.request(guard)
        self.assertEqual(status,429);self.assertGreater(int(headers[b'retry-after']),0)
        self.assertEqual(guard.active,0)

    def test_global_limit_and_bounded_identity_map(self):
        admission=Admission(clock=lambda:0,max_clients=10)
        self.assertTrue(all(admission.check(str(i),True)==0 for i in range(6)))
        self.assertGreater(admission.check('seventh',True),0)
        for i in range(100):admission.check(str(i),False)
        self.assertLessEqual(len(admission.clients),10)

    def test_refill_and_expiry(self):
        now=[0];a=Admission(clock=lambda:now[0],max_clients=1)
        for _ in range(3):self.assertEqual(a.check('one',True),0)
        self.assertGreater(a.check('one',True),0)
        now[0]=6;self.assertEqual(a.check('one',True),0)
        now[0]=607;self.assertEqual(a.check('two',True),0)

    def test_trust_only_tunnel_peer_and_group_ipv6(self):
        h={b'cf-connecting-ip':b'1.2.3.4',b'x-forwarded-for':b'9.9.9.9'}
        self.assertEqual(client_identity({'client':('8.8.8.8',1)},h),'8.8.8.8')
        self.assertEqual(client_identity({'client':('127.0.0.1',1)},h),'1.2.3.4')
        self.assertEqual(client_identity({'client':('2001:db8::123',1)},{}),'2001:db8::/64')

    async def test_concurrency_admission(self):
        guard=Protection(self.ok);guard.active=24
        self.assertEqual((await self.request(guard,path='/'))[0],405)
        self.assertEqual((await self.request(guard,path='/',method='GET'))[0],503)

    async def test_secure_response_and_method(self):
        guard=Protection(self.ok)
        status,h=await self.request(guard)
        self.assertEqual(status,200);self.assertEqual(h[b'cache-control'],b'no-store')
        self.assertEqual(h[b'x-frame-options'],b'DENY')
        self.assertEqual((await self.request(guard,method='TRACE'))[0],405)

if __name__=='__main__':unittest.main()
