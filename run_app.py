import os
import webbrowser
import asyncio


# RUNSERVER
async def runserver():
    path = "C:/Users/Admin/github/indivisible_goods_allocation/allocation_app"
    os.chdir(path)
    os.system("py manage.py runserver")

# OPEN BROWSER
def openproject():
    webbrowser.open_new_tab("http://127.0.0.1:8000/")

# EXECUTE PROGRAM
async def main():
    task1 = asyncio.create_task(runserver())
    openproject()
    await task1



asyncio.run(main())