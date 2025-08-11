# RCCL Tuner implementation.

Example RCCL tuner. It can be edited to suit different interconnects including Slingshot on Frontier or LUMI. The tuner alters the default selections made by RCCL, namely the algorithm and protocol used.

The `tuner.h` file comes directly from RCCL repository. The current implementation is forcing a tree algorithm for all-reduce collectives smaller than or equal to 64 bytes.

Assuming that the variable `ROCM_PATH` is set in your environment, pointing to where ROCm is installed, all you need to do to build is:
```
make
```
The result will be a shared library named `librccl-tuner.so`.

There is an example Makefile for LUMI that builds against the RCCL version available in the singularity containers. To use it you can run:
```
make -f Makefile.lumi
```

In order for RCCL to pick up the tuner you should set he environment variable:
```
export NCCL_TUNER_PLUGIN=<where you built the plugin>/librccl-tuner.so
```

If you enable logging with `export NCCL_DEBUG=INFO` you should see the plugin being picked up and the custom messages comming from it.
