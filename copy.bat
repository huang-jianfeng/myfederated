@echo off

@REM echo %date%
@REM echo %time%
@REM set srcpath="D:\Users\test\pythoncode\FL-bench\out\StreamingFedAvg"
@REM set destpath="D:\Users\test\pythoncode\FL-bench\out\StreamingFedAvg_%date:~0,4%_%date:~5,2%_%date:~8,2%_%time:~0,2%_%time:~3,2%_%time:~6,2%"
@REM xcopy %srcpath% %destpath% /E /I /Y
@REM timeout /t 3
@REM @REM set destpath="D:\Users\test\pythoncode\FL-bench\out\StreamingFedAvg_%date:~0,4%_%date:~5,2%_%date:~8,2%_%time:~0,2%_%time:~3,2%_%time:~6,2%"
@REM xcopy %srcpath% %destpath% /E /I /Y
cd /d "D:\Users\test\pythoncode\FL-bench\src\server"
call:myfunc

goto:eof
echo definestart
:myfunc
echo this is call
goto:eof
echo defineend