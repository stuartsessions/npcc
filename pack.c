
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
#include <stdbool.h>


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
// A utility function to reverse a string
void reverse(char str[], int length)
{
    int start = 0;
    int end = length - 1;
    while (start < end) {
        char temp = str[start];
        str[start] = str[end];
        str[end] = temp;
        end--;
        start++;
    }
}
// Implementation of citoa()
char* itoa(int num, char* str, int base)
{
    int i = 0;
    bool isNegative = false;
 
    /* Handle 0 explicitly, otherwise empty string is
     * printed for 0 */
    if (num == 0) {
        str[i++] = '0';
        str[i] = '\0';
        return str;
    }
 
    // In standard itoa(), negative numbers are handled
    // only with base 10. Otherwise numbers are
    // considered unsigned.
    if (num < 0 && base == 10) {
        isNegative = true;
        num = -num;
    }
 
    // Process individual digits
    while (num != 0) {
        int rem = num % base;
        str[i++] = (rem > 9) ? (rem - 10) + 'a' : rem + '0';
        num = num / base;
    }
 
    // If number is negative, append '-'
    if (isNegative)
        str[i++] = '-';
 
    str[i] = '\0'; // Append string terminator
 
    // Reverse the string
    reverse(str, i);
 
    return str;
}

/* The pond is a 2D array of cells */
static struct Cell pond[POND_SIZE_X][POND_SIZE_Y];
#define GENOME_SIZE 4096
static void readCell(FILE *file){
    char genomeData[GENOME_SIZE];
	if (fgets(genomeData, GENOME_SIZE, file) == NULL) {
		printf("Failed to read genome data from file\n");
		fclose(file);
		return;
	}
    int genomeIndex = 0;
	int bitIndex = 0;
	uintptr_t packedValue = 0;
    struct Cell cell;   
	for (int i = 0; genomeData[i] != '\0'; i++) {
		char character = genomeData[i];
		if (character == '0' || character == '1') {
			packedValue |= (character - '0') << bitIndex;
			bitIndex++;

			if (bitIndex == sizeof(uintptr_t) * 8) {
				cell.genome[genomeIndex] = packedValue;
				genomeIndex++;
				bitIndex = 0;
				packedValue = 0;
			}
		}
	}

	if (bitIndex > 0) {
		cell.genome[genomeIndex] = packedValue;
	}   
}

static void writeCell(FILE *file, struct Cell *cell) {
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
            char buffer [33];
            itoa(inst, buffer, 2);
            fprintf(file, "%s", buffer);
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
    //fprintf(file, "Hello, world!\n");

    // Call the readCell function
    struct Cell cell;
    cell.ID = 0;
    cell.parentID = 0;
    cell.lineage = 0;
    cell.generation = 0;
    cell.energy = 0;
    for(unsigned int i=0;i<POND_DEPTH_SYSWORDS;++i){
        cell.genome[i] = ~((uintptr_t)0);
    }

    //    fprintf(file, "%x\n", (unsigned int)cell.genome[i]);
   // }
    writeCell(file, &cell);
    //readCell(file);
    // Close the file
    fclose(file);

    return 0;
    }

