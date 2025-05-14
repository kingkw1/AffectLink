# filepath: app/ipc_manager.py
"""
Defines a SyncManager for sharing queues between processes (detector and dashboard).
"""
from multiprocessing.managers import SyncManager

# Create a subclass of SyncManager
class QueueManager(SyncManager):
    pass

# Register shared queue names
QueueManager.register('get_emotion_queue')
QueueManager.register('get_frame_queue')

# Helper to start the manager server

def start_manager(addr='127.0.0.1', port=50000, authkey=b'secret'):
    """
    Start a manager server exposing two shared queues: emotion_queue and frame_queue.
    """
    manager = QueueManager(address=(addr, port), authkey=authkey)
    manager.start()
    # Create the actual queues via manager
    emotion_queue = manager.get_emotion_queue()
    frame_queue = manager.get_frame_queue()
    return manager, emotion_queue, frame_queue

# Helper to connect to an existing manager server

def connect_manager(addr='127.0.0.1', port=50000, authkey=b'secret'):
    """
    Connect to a running manager server and return queue proxies.
    """
    manager = QueueManager(address=(addr, port), authkey=authkey)
    manager.connect()
    emotion_queue = manager.get_emotion_queue()
    frame_queue = manager.get_frame_queue()
    return manager, emotion_queue, frame_queue
