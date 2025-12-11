# 1. Project Configuration
DOCNAME = prelim
BUILD_DIR = build

# 2. Main Build Target
all:
	# Create the build directory if it doesn't exist
	mkdir -p $(BUILD_DIR)
	# Compile, sending all junk to the build dir
	latexmk -pdf -outdir=$(BUILD_DIR) $(DOCNAME).tex
	# Copy the final PDF back to the main folder so you can see it
	cp $(BUILD_DIR)/$(DOCNAME).pdf .

# 3. Watch Target (Live Updates)
watch:
	mkdir -p $(BUILD_DIR)
	latexmk -pvc -pdf -outdir=$(BUILD_DIR) $(DOCNAME).tex

# 4. Clean Target
clean:
	# Now we can just delete the whole folder!
	rm -rf $(BUILD_DIR)
	rm -f $(DOCNAME).pdf
