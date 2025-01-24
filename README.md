# Guide
## Before we get started
It is required to have the following installed on the pc:
```bash
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt install -y python3
sudo apt install -y python3-pip
sudo apt install -y python3-venv
sudo apt install -y make
```
## Setup Environment
To get started, a virtual environment needs to be created to allow for a coherent runtime experience separated from the software on your actual machine. Bellow must be run in order to build, activate and install the dependencies required for this software.
```bash 
make build-venv
source env/bin/activate
make dependencies
```
When you are finished working in the virtual environment, you can simply type the following to exit it:
```bash
deactivate
```

## Running the program
To run the program, you can simply type `make` in the terminal, and the appearing menu will guide you through the rest.

## Testing
All that you need to do is run:
```bash
make test
```
However, if you would like to run things manually one step at a time, below are some instructions on how to do so.

To test an individual unittest file, you can run the following command:
```bash
python3 -m coverage run <path/to/file>
```
A new file called _.coverage_ would have been made. To see the results in your terminal, type the following:
```bash
python3 -m coverage report
```
But if you rather want an interactive HTML experience, type the following:
```bash
python3 -m coverage html
```
This will generate an HTML file that can be opened in a browser for viewing.

To run all tests that are in the folder, run the following command:
```bash
python3 -m coverage run -m unittest discover <path/to/file>
```