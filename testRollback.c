#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BUFFER_SIZE 1000  // Size of the circular buffer
static uintptr_t buffer[BUFFER_SIZE];
static int in = 0;
static uintptr_t last_random_number;

volatile uint64_t prngState[2];

static inline uintptr_t getRandomPre()
{
	// https://en.wikipedia.org/wiki/Xorshift#xorshift.2B
	uint64_t x = prngState[0];
	const uint64_t y = prngState[1];
	prngState[0] = y;
	x ^= x << 23;
	const uint64_t z = x ^ y ^ (x >> 17) ^ (y >> 26);
	prngState[1] = z;
	return (uintptr_t)(z + y);
}
void precalculate_random_numbers() {
    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer[i] = getRandomPre();
    }
}
static inline uintptr_t getRandom() {
    uintptr_t num = buffer[in];
    last_random_number = num;  // Store the last random number
	buffer[in] = getRandomPre();  // Generate a new random number and add it to the buffer
    in = (in + 1) % BUFFER_SIZE;  // Wrap around to 0 when index reaches BUFFER_SIZE
    return num;
}
static inline uintptr_t getRandomRollback(uintptr_t rollback) {
    uintptr_t num = buffer[in];
    last_random_number = num;  // Store the last random number
    buffer[in] = getRandomPre();  // Generate a new random number and add it to the buffer
    in = ((in + 1) % BUFFER_SIZE) * rollback + in * (!rollback);  // Roll back if rollback is zero
    return num;
}


int main() {
    // Initialize the PRNG state
    prngState[0] = 13;
    prngState[1] = 0xDEADBEEF;

    // Precalculate random numbers
    precalculate_random_numbers();

    for (int i = 0; i < 2000; ++i) {
        uintptr_t numrollback = getRandomRollback(1);  
        //uintptr_t numpre = getRandomPre();
        //uintptr_t num = getRandom();
        print("num: %lu, numpre: %lu, numrollback: %lu\n", num, numpre, numrollback);
    }

    
    return 0;
}