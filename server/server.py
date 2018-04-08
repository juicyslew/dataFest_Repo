from aiohttp import web
import socketio
from time import sleep
# from server.tensorflow_model.NeuralNetworkObject import NN_Model


# neural_net = NN_Model()

sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

jobTypes = [
    'mednurse',
    'doctor',
    'hippie',
]


async def index(request):
    with open('../index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')


@sio.on('connect')
def connect(sid, environ):
    sleep(0.5)
    sio.emit('message', jobTypes, room=sid)
    print('Connected')


@sio.on('gimmeData')
async def message(sid):
    await sio.emit('jobTypes', jobTypes, room=sid)


@sio.on('request')
async def message(sid, data):
    print("request")
    try:
        jobType = data['jobType']
        experience = data['experience']
        edLevel = data['edLevel']
        isSupervisorRole = data['supervisor']
    except KeyError:
        print('ERROR: Query missing data')
        print(data)
        return
    print(data)
    # neural_net.HeatMap()


@sio.on('disconnect')
def disconnect(sid):
    print('Client disconnected')


# app.router.add_static('/static', 'static')
# app.router.add_get('/', index)


if __name__ == '__main__':
    web.run_app(app, port=1234)
