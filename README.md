The engines used for benchmarking have been created by Ed Gilbert and Martin Fierz. All credit goes to them.

All credit for Cake goes to Martin Fierz. The function python_getmove has been added to [cakepp.c](./cake-1.20/cakepp.c) in order to interface with python.
Cake's bindings are only configured for Linux. Potentially you will have to recompile. Go to [cake-1.20](./cake-1.20) and use
```
gcc -fPIC -shared db.c movegen.c ansicake.c book.c cakepp.c -o cakepp.o
```