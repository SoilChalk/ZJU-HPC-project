CC = gcc
CFLAGS = -std=c11 -O3 -fopenmp
TARGET = winograd

# Point to the final source directory
SRC_DIR = src
SOURCES = $(SRC_DIR)/main.c $(SRC_DIR)/naive_conv.c $(SRC_DIR)/winograd_conv.c

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: clean