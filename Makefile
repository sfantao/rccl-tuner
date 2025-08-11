ROCM_PATH ?= /opt/rocm
RCCL_HOME ?= $(ROCM_PATH)/rccl

INC = -D__HIP_PLATFORM_AMD__ -I$(RCCL_HOME)/include -I$(ROCM_PATH)/include

LDFLAGS =

PLUGIN_SO = librccl-tuner.so

all: $(PLUGIN_SO)

$(PLUGIN_SO): tuner.c tuner.h
	$(CC) $(INC) -O3 -ggdb -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) tuner.c

clean:
	rm -f $(PLUGIN_SO)
