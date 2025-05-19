import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need
# fine tuning.
build_exe_options = {
    "packages": ["flask", "jinja2", "werkzeug"],  # Add any other packages your app requires
    "include_files": ["templates", "static", "Attendance"]  # Add any additional directories or files your app needs
}

# GUI applications require a different base on Windows (the default is for a console application).
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="attendanceSystem",  # Replace "attendanceSystem" with the name of your application
    version="1.0",
    description="Attendance System Description",
    options={"build_exe": build_exe_options},
    executables=[Executable("app.py", base=base)]
)
