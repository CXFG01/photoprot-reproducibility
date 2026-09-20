"""Run on the existing Brev environment with a fake engine, no model load."""
import asyncio
from collections import OrderedDict
import threading
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from webapp import server


class Upload:
    def __init__(self,chunks=(b'image',),delay=0):self.chunks=chunks;self.delay=delay
    async def stream(self):
        for chunk in self.chunks:
            if self.delay:await asyncio.sleep(self.delay)
            yield chunk


class Engine:
    def search(self,data):return {'results':[]}


class ServerSafetyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        s=server.app.state;s.pending=0;s.gpu=asyncio.Lock();s.engine=Engine()
        s.exports=OrderedDict();s.metadata=OrderedDict();s.metadata_tasks={}
        s.structures=OrderedDict();s.structure_bytes=0;s.structure_tasks={}
        s.allowed_pdbs={'3I3W','1UBQ'}

    async def test_slow_and_chunked_upload_rejected(self):
        with patch.object(server,'UPLOAD_TIMEOUT',.01):
            with self.assertRaises(HTTPException) as e:await server.search(Upload(delay=.1))
            self.assertEqual(e.exception.status_code,408)
        with patch.object(server,'MAX_BYTES',3):
            with self.assertRaises(HTTPException) as e:await server.search(Upload((b'12',b'34')))
            self.assertEqual(e.exception.status_code,413)
        self.assertEqual(server.app.state.pending,0)

    async def test_cancellation_does_not_overlap_gpu(self):
        entered=threading.Event();release=threading.Event()
        class SlowEngine:
            def search(self,data):entered.set();release.wait(3);return {'results':[]}
        server.app.state.engine=SlowEngine()
        task=asyncio.create_task(server.search(Upload()))
        for _ in range(100):
            if entered.is_set():break
            await asyncio.sleep(.01)
        self.assertTrue(entered.is_set());task.cancel();await asyncio.sleep(.02)
        self.assertTrue(server.app.state.gpu.locked());self.assertEqual(server.app.state.pending,1)
        release.set()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertFalse(server.app.state.gpu.locked());self.assertEqual(server.app.state.pending,0)

    async def test_queue_deadline_and_capacity(self):
        server.app.state.pending=3
        with self.assertRaises(HTTPException) as e:await server.search(Upload())
        self.assertEqual(e.exception.status_code,429)
        server.app.state.pending=0;await server.app.state.gpu.acquire()
        try:
            with patch.object(server,'QUEUE_TIMEOUT',.01):
                with self.assertRaises(HTTPException) as e:await server.search(Upload())
                self.assertEqual(e.exception.status_code,503)
        finally:server.app.state.gpu.release()
        self.assertEqual(server.app.state.pending,0)

    async def test_outbound_coalescing_and_allowlist(self):
        calls=[]
        async def fake(pdb):calls.append(pdb);await asyncio.sleep(.01);return {'pdb_id':pdb}
        with patch.object(server,'fetch_metadata',fake):
            await asyncio.gather(*(server.metadata('3I3W') for _ in range(20)))
        self.assertEqual(calls,['3I3W'])
        with self.assertRaises(HTTPException) as e:await server.entries('ZZZZ')
        self.assertEqual(e.exception.status_code,404)
        with self.assertRaises(HTTPException) as e:await server.structure('ZZZZ')
        self.assertEqual(e.exception.status_code,404)

    async def test_structure_cache_bounded(self):
        server.app.state.allowed_pdbs.update({'0001','0002','0003','0004','0005','0006'})
        async def fake(pdb):return b'x'*(12*1024*1024)
        with patch.object(server,'fetch_structure',fake):
            for pdb in ('0001','0002','0003','0004','0005','0006'):await server.structure(pdb)
        self.assertLessEqual(server.app.state.structure_bytes,64*1024*1024)
        self.assertNotIn('0001',server.app.state.structures)

if __name__=='__main__':unittest.main()
