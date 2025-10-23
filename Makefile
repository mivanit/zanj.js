.PHONY: test clean clean-temp generate-test-data run-tests help

# Default target
help:
	@echo "Available targets:"
	@echo "  make test              - Run all tests (clean, generate data, run tests)"
	@echo "  make generate-test-data - Generate Python test data"
	@echo "  make run-tests         - Run JavaScript tests"
	@echo "  make clean            - Remove generated test data"
	@echo "  make clean-temp       - Remove .temp directory"

# Run all tests
test: clean-temp generate-test-data run-tests

# Clean up generated test data
clean-temp:
	@echo "Cleaning tests/.temp directory..."
	@rm -rf tests/.temp

# Generate test data using Python
generate-test-data:
	@echo "Generating test data..."
	@mkdir -p tests/.temp
	@uv run tests/generate-test-data.py

# Run JavaScript tests
run-tests:
	@echo "\n=========================================="
	@echo "Running JavaScript tests..."
	@echo "==========================================\n"
	@node --test tests/test-inline-arrays.js
	@node --test tests/test-python-generated.js
	@echo "\n=========================================="
	@echo "All tests completed!"
	@echo "==========================================\n"

# Alias for clean-temp
clean: clean-temp
