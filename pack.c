
#define POND_SIZE_X 800
#define POND_SIZE_Y 600

/* Depth of pond in four-bit codons -- this is the maximum
 * genome size. This *must* be a multiple of 16! */
#define POND_DEPTH 1024


/* ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


/* Pond depth in machine-size words.  This is calculated from
 * POND_DEPTH and the size of the machine word. (The multiplication
 * by two is due to the fact that there are two four-bit values in
 * each eight-bit byte.) */
#define POND_DEPTH_SYSWORDS (POND_DEPTH / (sizeof(uintptr_t) * 2))

/* Number of bits in a machine-size word */
#define SYSWORD_BITS (sizeof(uintptr_t) * 8)

/* Word and bit at which to start execution */
/* This is after the "logo" */
#define EXEC_START_WORD 0
#define EXEC_START_BIT 4

/* Number of bits set in binary numbers 0000 through 1111 */
static const uintptr_t BITS_IN_FOURBIT_WORD[16] = { 0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4 };

/**
 * Structure for a cell in the pond
 */
struct Cell
{
	/* Globally unique cell ID */
	uint64_t ID;
	
	/* ID of the cell's parent */
	uint64_t parentID;
	
	/* Counter for original lineages -- equal to the cell ID of
	 * the first cell in the line. */
	uint64_t lineage;
	
	/* Generations start at 0 and are incremented from there. */
	uintptr_t generation;
	
	/* Energy level of this cell */
	uintptr_t energy;

	/* Memory space for cell genome (genome is stored as four
	 * bit instructions packed into machine size words) */
	uintptr_t genome[POND_DEPTH_SYSWORDS];
};

/* The pond is a 2D array of cells */
static struct Cell pond[POND_SIZE_X][POND_SIZE_Y];

static void readCell(FILE *file, struct Cell *cell) {
    uintptr_t wordPtr,shiftPtr,inst,stopCount,i;
		wordPtr = 0;
		shiftPtr = 0;
		stopCount = 0;
		for(i=0;i<POND_DEPTH;++i) {
			inst = (cell->genome[wordPtr] >> shiftPtr) & 0xf;
			/* Four STOP instructions in a row is considered the end.
			 * The probability of this being wrong is *very* small, and
			 * could only occur if you had four STOPs in a row inside
			 * a LOOP/REP pair that's always false. In any case, this
			 * would always result in our *underestimating* the size of
			 * the genome and would never result in an overestimation. */
			fprintf(file,"%x",(unsigned int)inst);
			if (inst == 0xf) { /* STOP */
				if (++stopCount >= 4)
					break;
			} else stopCount = 0;
			if ((shiftPtr += 4) >= SYSWORD_BITS) {
				if (++wordPtr >= POND_DEPTH_SYSWORDS) {
					wordPtr = EXEC_START_WORD;
					shiftPtr = EXEC_START_BIT;
				} else shiftPtr = 0;
			}
		}
	fprintf(file,"\n");
}
int main(int argc, char** argv) {
    FILE *file = fopen("file.txt", "w");
    if (file == NULL) {
        printf("Failed to create the file.\n");
        return 1;
    }

    // Write data to the file
    fprintf(file, "Hello, world!\n");

    // Call the readCell function
    struct Cell cell;
    cell.ID = 0;
    cell.parentID = 0;
    cell.lineage = 0;
    cell.generation = 0;
    cell.energy = 0;
    for(unsigned int i=0;i<POND_DEPTH_SYSWORDS-1;++i){
        cell.genome[i] = ~((uintptr_t)0);
    }
    for(unsigned int i=POND_DEPTH_SYSWORDS;i<POND_DEPTH_SYSWORDS;++i){
        cell.genome[i] = 0;
    }
    for(unsigned int i=0;i<POND_DEPTH_SYSWORDS;++i){
        fprintf(file, "%x\n", (unsigned int)cell.genome[i]);
    }
    readCell(file, &cell);

    // Close the file
    fclose(file);

    return 0;
    }

