#define GENOME_SIZE 4096

void packGenome(struct Cell cell, const char* filename) {
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		printf("Failed to open file: %s\n", filename);
		return;
	}

	char genomeData[GENOME_SIZE];
	if (fgets(genomeData, GENOME_SIZE, file) == NULL) {
		printf("Failed to read genome data from file\n");
		fclose(file);
		return;
	}

	fclose(file);

	int genomeIndex = 0;
	int bitIndex = 0;
	uintptr_t packedValue = 0;

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
