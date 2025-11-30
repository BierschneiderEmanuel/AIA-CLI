# AIA-CLI

<p align="center">
  <img src="logo.png" width="259" height="121" alt="Project Logo">
</p>

AI Advanced CLI

# Installation
sudo apt-get install dos2unix
pip install -r requirements.txt
sudo cp ai.py /usr/local/bin/aia && sudo dos2unix /usr/local/bin/aia

# Show help
aia --help

# Example Prompt "Hello"
echo "Hello" | aia --model llama3.1:8b --print

# Python documentation string generation
for file in ./*.py; do 
   aia -p "generate docstring for $file" --output-format text >> docs.md
done

# Pull request review
git diff origin/main...HEAD | aia --model llama3.1:8b --pr_review

# Unit Test
for file in ./*.py; do 
   echo "Generating tests for $file..."
   aia -m "llama3.3:70b" -p "You are a Python testing expert specializing in edge case testing. Your task is to write Python test code to thoroughly test all functions in the provided file. The test code must include: 1. Happy path scenarios for each function. 2. Edge cases, including unexpected inputs such as None, -1, 0, 1, empty lists ([]), and invalid data types. 3. Error path scenarios, ensuring the functions handle exceptions gracefully. 4. Comprehensive assertions to validate the correctness of the function outputs. 5. Tests for asynchronous functions, if applicable. The test code must be written in Python using the pytest framework. Ensure that the test cases are modular, well-documented, and include setup and teardown methods if necessary. Do not include any explanations or comments outside the test code itself. Only output the Python test code. Always test your test code using the code tool to execute the python code. This is the code you should test: $(cat "$file")" >> tests.py
done

# Dump vectorstore content
echo "Dump" | aia --model llama3.1:8b --dump

# Search vectorstore content
echo "Dump" | aia --model llama3.1:8b --search "lorem ipsum"


