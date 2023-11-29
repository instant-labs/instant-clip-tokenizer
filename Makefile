# This will only work on macOS, for other operating systems the filename below
# needs to be adjusted, see
# https://pyo3.rs/v0.20.0/building_and_distribution.html#manual-builds
test-python:
	cargo build --release
	cp target/release/libinstant_clip_tokenizer.dylib instant-clip-tokenizer-py/test/instant_clip_tokenizer.so
	PYTHONPATH=instant-clip-tokenizer-py/test/ python3 -m test
