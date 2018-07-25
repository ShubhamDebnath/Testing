from sys import platform
extensions = {
	"c" : 'c' ,
	"c++" : 'cpp',
	"c#" : 'cs',
	"java" : 'java',
	"javascript" : 'js',
	"python3" : 'py'
}

compilers = {
	"c" : 'gcc' ,
	"c++" : 'g++',
	"c#" : 'csc',
	"java" : 'javac',
	"javascript" : 'node',
	"python3" : 'python'
}

def_codes = {
	
	"c":
	"""
// A Hello World! program in C.
#include<stdio.h>

	int main()
	{
		printf("Hello World !");
		return 0;
	}
	""" ,


	"c++":
	'''
// A Hello World! program in C++.
#include<iostream.h>
using namespace std;

	int main()
	{
		cout<<"Hello World !";
		return 0;
	}
	''',


	"c#":
	'''
// A Hello World! program in C#.
using System;
namespace HelloWorld
{
    class Hello 
    {
        static void Main() 
        {
            Console.WriteLine("Hello World !");
        }
    }
}
	''',


	"java":
	"""
// A Hello World! program in JAVA.
public class Main
{
	public static void main(String args[])
	{
		System.out.println("Hello World !");
	}
}
	""",


	"javascript":
	"""
// A Hello World! program in JAVASCRIPT.
console.log("Hello World !");
	""",


	"python3":
	"""
# A Hello World! program in PYTHON3.
print("Hello World !")
	"""
}

def return_default_code(choice):
	return def_codes[choice]

def return_extension(choice):
	return extensions[choice]

def return_compiler(choice):
	return compilers[choice]

def return_os():
	if platform == "linux" or platform == "linux2":
		command_separator = '|'
	elif platform == "darwin":
		command_separator = '&&'
	elif platform == "win32":
		command_separator = '&&'

	return command_separator

if __name__ == '__main__':
	print(return_default_code("C++"))