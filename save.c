
#include <stdio.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include "shared.h"

static struct Cell readCell(char *genomeData) {
    uintptr_t wordPtr = 0;
    uintptr_t shiftPtr = 0;
    uintptr_t packedValue = 0;
    struct Cell cell;

    //printf("Size of genomeData: %zu\n", strlen(genomeData));
    
    for (int i = 0; genomeData[i] != '\0'; i++) {
        char character = genomeData[i];
        //printf("%c", character);
        if (character == '0' || character == '1') {
            packedValue |= (character - '0') << shiftPtr;
            shiftPtr += 4;

            if (shiftPtr >= SYSWORD_BITS) {
                if (wordPtr >= sizeof(cell.genome) / sizeof(cell.genome[0])) {
                    break; // Prevents memory error
                }
                cell.genome[wordPtr] = packedValue;
                wordPtr++;
                shiftPtr = 0;
                packedValue = 0;
            }
        }
    }

    if (shiftPtr > 0) {
        cell.genome[wordPtr] = packedValue;
    }
    return cell;
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
            fprintf(file, "%x", inst);
			if (inst == 0xf) { /* STOP */
				if (++stopCount >= 4)
                    fprintf(file, "%s", "stopped");
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