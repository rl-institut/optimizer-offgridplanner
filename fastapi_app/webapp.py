import os
import json
import io
from fastapi import FastAPI, Request, Response, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse

try:
    from worker import app as celery_app
except ModuleNotFoundError:
    from .worker import app as celery_app
import celery.states as states

app = FastAPI()

SERVER_ROOT = os.path.dirname(__file__)

app.mount(
    "/static", StaticFiles(directory=os.path.join(SERVER_ROOT, "static")), name="static"
)

templates = Jinja2Templates(directory=os.path.join(SERVER_ROOT, "templates"))


# option for routing `@app.X` where `X` is one of
# post: to create data.
# get: to read data.
# put: to update data.
# delete: to delete data.

# while it might be tempting to use BackgroundTasks for oemof simulation, those might take up
# resources and it is better to start them in an independent process. BackgroundTasks are for
# not resource intensive processes(https://fastapi.tiangolo.com/tutorial/background-tasks/)


# `127.0.0.1:8000/docs` endpoint will have autogenerated docs for the written code

# Test Driven Development --> https://fastapi.tiangolo.com/tutorial/testing/


@app.get("/")
def index(request: Request) -> Response:

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        },
    )


async def simulate_json_variable(request: Request, queue: str = "supply"):
    """Receive mvs simulation parameter in json post request and send it to simulator"""
    input_dict = await request.json()

    # send the task to celery
    task = celery_app.send_task(
        f"{queue}.run_simulation", args=[input_dict], queue=queue, kwargs={}
    )
    queue_answer = await check_task(task.id)

    return queue_answer


@app.post("/sendjson/grid")
async def simulate_json_variable_grid(request: Request):
    return await simulate_json_variable(request, queue="grid")


@app.post("/sendjson/supply")
async def simulate_json_variable_supply(request: Request):
    return await simulate_json_variable(request, queue="supply")


@app.post("/uploadjson/grid")
def simulate_uploaded_json_files_grid(
    request: Request, json_file: UploadFile = File(...)
):
    """Receive mvs simulation parameter in json post request and send it to simulator
    the value of `name` property of the input html tag should be `json_file` as the second
    argument of this function
    """
    json_content = jsonable_encoder(json_file.file.read())
    return run_simulation(request, input_json=json_content, queue="grid")


@app.post("/uploadjson/supply")
def simulate_uploaded_json_files_supply(
    request: Request, json_file: UploadFile = File(...)
):
    """Receive mvs simulation parameter in json post request and send it to simulator
    the value of `name` property of the input html tag should be `json_file` as the second
    argument of this function
    """
    json_content = jsonable_encoder(json_file.file.read())
    return run_simulation(request, input_json=json_content, queue="supply")


def run_simulation(request: Request, input_json=None, queue="supply") -> Response:
    """Send a simulation task to a celery worker"""

    if input_json is None:
        input_dict = {
            "name": "dummy_json_input",
            "secondary_dict": {"val1": 2, "val2": [5, 6, 7, 8]},
        }
    else:
        input_dict = json.loads(input_json)

    # send the task to celery
    task = celery_app.send_task(
        f"{queue}.run_simulation", args=[input_dict], queue=queue, kwargs={}
    )

    return templates.TemplateResponse(
        "submitted_task.html", {"request": request, "task_id": task.id}
    )



@app.get("/check/{task_id}")
async def check_task(task_id: str) -> JSONResponse:
    res = celery_app.AsyncResult(task_id)
    task = {
        "server_info": None,
        "id": task_id,
        "status": res.state,
        "results": None,
    }
    if res.state == states.PENDING:
        task["status"] = res.state
    else:
        task["status"] = "DONE"
        results_as_dict = json.loads(res.result)
        server_info = results_as_dict.pop("SERVER")
        task["server_info"] = server_info
        task["results"] = results_as_dict
        if "ERROR" in task["results"]:
            task["status"] = "ERROR"
            task["results"] = results_as_dict

    return JSONResponse(content=jsonable_encoder(task))


@app.get("/abort/{task_id}")
async def revoke_task(task_id: str) -> JSONResponse:
    res = celery_app.AsyncResult(task_id)
    res.revoke(terminate=True)
    return JSONResponse(content=jsonable_encoder({"task_id": task_id, "aborted": True}))
