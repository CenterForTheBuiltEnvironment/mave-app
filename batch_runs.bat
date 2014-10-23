echo off
setlocal EnableDelayedExpansion

SET LOGFILE=batch.log

SET F= Main_cbe_01.csv
SET N= 3
SET K= 2 5
SET R= 0.1

FOR %%f in (%F%) DO ( 
    FOR %%n in (%N%) DO ( 
        FOR %%k in (%K%) DO (
            FOR %%r in (%R%) DO (
                Echo params: %%f -n %%n -k %%k -r %%r > %LOGFILE%
                python bep.py %%f -c 0.01 -v -n 14 -nv %%n -pf 0.33 -k %%k -rs %%r -s >> %LOGFILE%  
            )
        )
    )
)
pause
