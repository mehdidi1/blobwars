# Remove optimize_weights.o and ga_weight_optimizer.o from OBJS
OBJS = strategy.o blobwar.o main.o font.o mouse.o image.o widget.o rollover.o button.o label.o board.o rules.o blob.o network.o bidiarray.o shmem.o greedy.o minimax.o minimax_para.o alpha_beta.o alpha_beta_para.o feldman.o final_ai.o test.o

OBJS_launchComputation = launchStrategy.o strategy.o bidiarray.o shmem.o greedy.o minimax.o \
                    	minimax_para.o alpha_beta.o alpha_beta_para.o feldman.o final_ai.o test.o

# Better approach: include all the AI modules the optimizer might need
OBJS_optimizer = optimize_weights.o ga_weight_optimizer.o strategy.o bidiarray.o \
                 alpha_beta.o alpha_beta_para.o greedy.o minimax.o minimax_para.o feldman.o

LIBS = -lSDL_image -lSDL_ttf -lm `sdl-config --libs` -lSDL_net -lpthread -ltbb

CFLAGS = -Wall -Werror -O3 -g `sdl-config --cflags`  -Wno-strict-aliasing -DDEBUG -DUSE_MINIMAX
CC = g++

# $(sort) remove duplicate object
OBJS_ALL = $(sort $(OBJS) $(OBJS_launchComputation) $(OBJS_optimizer))

blobwar: $(OBJS) launchStrategy
	$(CC) $(OBJS) $(CFLAGS) -o blobwar $(LIBS)

$(OBJS_ALL):	%.o:	%.cc
	$(CC) -c $<  $(CFLAGS)

launchStrategy: $(OBJS_launchComputation)
	$(CC) $(OBJS_launchComputation) $(CFLAGS) -o launchStrategy $(LIBS)

optimize_weights: $(OBJS_optimizer)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f *.o core blobwar launchStrategy optimize_weights *~
