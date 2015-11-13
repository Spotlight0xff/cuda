# Location of the CUDA Toolkit
CUDA_PATH ?= "/opt/cuda"

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= x86_64
TARGET_SIZE := 64

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
PROG 		  := vectorAdd
SOURCES       := vectorAdd.cu
OBJECTS 	  := $(SOURCES:.cu=.o)




# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     :=
LDFLAGS     :=

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I../../common/inc
LIBRARIES :=

################################################################################


# Gencode arguments
SMS ?= 20 30 35 37 50

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif


################################################################################

# Target rules
all: build

build: $(PROG)


%.o: %.cu
	mkdir -p ../obj/
	 $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o ../obj/$@ -c $<

$(PROG): $(OBJECTS)
	@echo $(OBJECTS)
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ ../obj/$+ $(LIBRARIES)
	mkdir -p ../bin/
	mv $@ ../bin/


run: build
	../bin/$(PROG)

clean:
	cd ../obj/ && rm -f $(OBJECTS)
	rm -rf ../bin/$(PROG)

clobber: clean
