#include<stdio.h>
#include<unistd.h>

int main(){
    int pid, i, loopsize;
    
    loopsize = 1000;
    pid = fork();
    
    if(pid == 0){
	// Child process
	for(i = 0; i < loopsize; i++){
	    printf("child: %d\n", i + 1);
	}
    }
    else{
	// Parent process
	for(i = 0; i < loopsize; i++){
	    printf("parent: %d\n", i + 1);
	}
    }
    return 0;
}
