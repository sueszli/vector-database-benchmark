/* SPDX-License-Identifier: BSD-2-Clause */

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/mman.h>

#include "devshell.h"
#include "sysenter.h"

/*
 * Call waitpid() after the child exited.
 */
int cmd_wpid1(int argc, char **argv)
{
   int wstatus;
   pid_t pid;

   int child_pid = fork();

   if (child_pid < 0) {
      printf("fork() failed\n");
      return 1;
   }

   if (!child_pid) {
      // This is the child, just exit
      printf("child: exit\n");
      exit(23);
   }

   printf("Created child with pid: %d\n", child_pid);

   /* Wait for the child to exit */
   usleep(100*1000);

   /* Now, let's see call waitpid() after the child exited */
   pid = waitpid(child_pid, &wstatus, 0);

   if (!WIFEXITED(wstatus)) {

      printf("[pid: %d] PARENT: the child %d did NOT exited normally\n",
             getpid(), pid);

      return 1;
   }

   int exit_code = WEXITSTATUS(wstatus);
   printf("waitpid() returned %d, exit code: %d\n", pid, exit_code);

   if (pid != child_pid) {
      printf("Expected waitpid() to return child's pid (got: %d)\n", pid);
      return 1;
   }

   if (exit_code != 23) {
      printf("Expected the exit code to be 23 (got: %d)\n", exit_code);
      return 1;
   }

   return 0;
}

/* waitpid(-1): wait any child to exit */
int cmd_wpid2(int argc, char **argv)
{
   const int child_count = 3;

   int pids[child_count];
   int wstatus;
   pid_t pid;

   for (int i = 0; i < child_count; i++) {

      int child_pid = fork();

      if (child_pid < 0) {
         printf("fork() failed\n");
         return 1;
      }

      if (!child_pid) {
         printf("[pid: %d] child: exit (%d)\n", getpid(), 10 + i);
         usleep((child_count - i) * 100*1000);
         exit(10 + i); // exit from the child
      }

      pids[i] = child_pid;
   }

   usleep(120 * 1000);

   for (int i = child_count-1; i >= 0; i--) {

      printf("[pid: %d] PARENT: waitpid(-1)\n", getpid());
      pid = waitpid(-1, &wstatus, 0);

      if (!WIFEXITED(wstatus)) {

         printf("[pid: %d] PARENT: the child %d did NOT exited normally\n",
                 getpid(), pid);

         return 1;
      }

      int exit_code = WEXITSTATUS(wstatus);

      printf("[pid: %d] PARENT: waitpid() returned %d, exit code: %d\n",
             getpid(), pid, exit_code);

      if (pid != pids[i]) {
         printf("Expected waitpid() to return %d (got: %d)\n", pids[i], pid);
         return 1;
      }

      if (exit_code != 10+i) {
         printf("Expected the exit code to be %d (got: %d)\n", 10+i, exit_code);
         return 1;
      }
   }

   return 0;
}

/* wait on any child after they exit */
int cmd_wpid3(int argc, char **argv)
{
   int wstatus;
   pid_t pid;

   int child_pid = fork();

   if (child_pid < 0) {
      printf("fork() failed\n");
      return 1;
   }

   if (!child_pid) {
      /* This is the child, just exit with a special code */
      printf("child: exit\n");
      exit(23);
   }

   printf("Created child with pid: %d\n", child_pid);

   /* Wait for the child to exit */
   usleep(100*1000);

   /* Now, let's see call waitpid() after the child exited */
   pid = waitpid(-1, &wstatus, 0);

   if (!WIFEXITED(wstatus)) {

      printf("[pid: %d] PARENT: the child %d did NOT exited normally\n",
             getpid(), pid);

      return 1;
   }

   int exit_code = WEXITSTATUS(wstatus);
   printf("waitpid() returned %d, exit code: %d\n", pid, exit_code);

   if (pid != child_pid) {
      printf("Expected waitpid() to return child's pid (got: %d)\n", pid);
      return 1;
   }

   if (exit_code != 23) {
      printf("Expected the exit code to be 23 (got: %d)\n", exit_code);
      return 1;
   }

   return 0;
}


/*
 * Test the case of a parent dying before its children.
 */
int cmd_wpid4(int argc, char **argv)
{
   pid_t pid;
   int wstatus;
   int child_pid;
   bool failed = false;

   printf("[grandparent] my pid: %d\n", getpid());

   child_pid = fork();

   if (child_pid < 0) {
      printf("fork() failed\n");
      return 1;
   }

   if (!child_pid) {

      /* in the child: now create other children and die before them */

      int grand_child1, grand_child2;

      grand_child1 = fork();

      if (grand_child1 < 0) {
         printf("fork() failed\n");
         exit(1);
      }

      if (!grand_child1) {
         usleep(50*1000);
         exit(10);
      }

      printf(STR_PARENT "child 1: %d\n", grand_child1);

      grand_child2 = fork();

      if (grand_child2 < 0) {
         printf("fork() failed\n");
         exit(1);
      }

      if (!grand_child2) {
         usleep(100*1000);
         exit(11);
      }

      printf(STR_PARENT "child 2: %d\n", grand_child2);
      printf(STR_PARENT "exit\n");
      exit(0);
   }

   /* in the grandparent: wait for any child */
   printf("[grandparent] child pid: %d\n", child_pid);

   while ((pid = waitpid(-1, &wstatus, 0)) > 0) {

      int code = WEXITSTATUS(wstatus);
      printf("[grandparent] waitpid(): child %d exited with %d\n", pid, code);

      if (code != 0)
         failed = true;
   }

   printf("[grandparent] exit (failed: %d)\n", failed);
   return failed ? 1 : 0;
}

void
print_waitpid_change(int child, int wstatus)
{
   int code = WEXITSTATUS(wstatus);
   int sig = WTERMSIG(wstatus);

   if (WIFSTOPPED(wstatus))
      printf(STR_PARENT "child %d: STOPPED\n", child);
   else if (WIFCONTINUED(wstatus))
      printf(STR_PARENT "child %d: CONTINUED\n", child);
   else if (WIFEXITED(wstatus))
      printf(STR_PARENT "child %d: EXITED with %d\n", child, code);
   else if (WIFSIGNALED(wstatus))
      printf(STR_PARENT "child %d: KILLED by sig: %d\n", child, sig);
   else
      printf(STR_PARENT "child %d: UNKNOWN status change!\n", child);

   fflush(stdout);
}

static int
call_waitpid(int wait_pid, int *wstatus, int *children, int n)
{
   int pid = waitpid(wait_pid, wstatus, WUNTRACED | WCONTINUED);

   if (pid > 0) {

      int child_index = -1;

      for (int i = 0; i < n && child_index < 0; i++)
         if (children[i] == pid)
            child_index = i;

      print_waitpid_change(child_index, *wstatus);
   }

   return pid;
}

/*
 * Wait on children getting SIGSTOP and SIGCONT
 */
int cmd_wpid5(int argc, char **argv)
{
   pid_t children[2];
   int pid, wstatus;

   children[0] = fork();

   if (children[0] < 0) {
      printf(STR_PARENT "fork() failed\n");
      return 1;
   }

   if (!children[0]) {

      /* child 0's body */
      printf("[child 0] Hello from pid %d\n", getpid());

      for (int i = 0; i < 10; i++) {
         printf("[child 0] i = %d\n", i);
         usleep(100 * 1000);
      }

      printf("[child 0] exit\n");
      exit(0);
   }

   printf(STR_PARENT "children[0] pid: %d\n", children[0]);

   children[1] = fork();

   if (children[1] < 0) {
      printf(STR_PARENT "fork() failed\n");
      kill(children[0], SIGKILL);
      return 1;
   }

   if (!children[1]) {

      /* child 1's body */
      printf("[child 1] Hello from pid %d\n", getpid());

      printf("[child 1] Wait some time...\n");
      usleep(250 * 1000);

      printf("[child 1] Send SIGSTOP to child 0\n");
      kill(children[0], SIGSTOP);

      printf("[child 1] Wait some time...\n");
      usleep(250 * 1000);

      printf("[child 1] Send SIGCONT to child 0\n");
      kill(children[0], SIGCONT);

      printf("[child 1] Wait some time...\n");
      usleep(50 * 1000);

      printf("[child 1] exit\n");
      exit(0);
   }

   /* Expect child 0 stopped */
   pid = call_waitpid(-1, &wstatus, children, 2);
   DEVSHELL_CMD_ASSERT(pid == children[0]);
   DEVSHELL_CMD_ASSERT(WIFSTOPPED(wstatus));

   /* Expect child 0 continued */
   pid = call_waitpid(-1, &wstatus, children, 2);
   DEVSHELL_CMD_ASSERT(pid == children[0]);
   DEVSHELL_CMD_ASSERT(WIFCONTINUED(wstatus));

   /* Expect that child 1 exited */
   pid = call_waitpid(-1, &wstatus, children, 2);
   DEVSHELL_CMD_ASSERT(pid == children[1]);
   DEVSHELL_CMD_ASSERT(WIFEXITED(wstatus));

   /* Expect that child 0 exited */
   pid = call_waitpid(-1, &wstatus, children, 2);
   DEVSHELL_CMD_ASSERT(pid == children[0]);
   DEVSHELL_CMD_ASSERT(WIFEXITED(wstatus));

   return 0;
}

static void
wpid6_test_child(int n, int pgid)
{
   printf("[child %d] Hello from pid %d\n", n, getpid());

   printf("[child %d] curr pgid: %d\n", n, getpgid(0));
   setpgid(0, pgid);
   printf("[child %d] new pgid: %d\n", n, getpgid(0));


   for (int i = 0; i < 10; i++) {
      printf("[child %d] i = %d\n", n, i);
      usleep(100 * 1000);
   }

   printf("[child %d] exit\n", n);
   exit(0);
}

static void
wpid6_test_active_child(int n, int pgid1, int pgid2)
{
   printf("[active child %d] Hello from pid %d\n", n, getpid());
   usleep(500 * 1000);

   printf("[active child %d] Send SIGKILL to the pgid %d\n", n, pgid1);
   kill(-pgid1, SIGKILL);

   usleep(100 * 1000);
   printf("[active child %d] Send SIGKILL to the pgid %d\n", n, pgid2);
   kill(-pgid2, SIGKILL);

   printf("[active child %d] exit\n", n);
   exit(0);
}

/*
 * Wait on a process group + send a signal to a process group
 */
int cmd_wpid6(int argc, char **argv)
{
   int cld[7] = {0};
   int pid, wstatus;
   int g1_killed = 0;
   int g2_killed = 0;

   printf(STR_PARENT "Hello, pid: %d, pgid: %d\n", getpid(), getpgid(0));
   printf(STR_PARENT "Start children..\n");

   for (int i = 0; i < ARRAY_SIZE(cld) - 1; i++) {

      cld[i] = fork();
      DEVSHELL_CMD_ASSERT(cld[i] >= 0);

      if (!cld[i]) {

         int pgid = i < 3 ? cld[0] : cld[3];
         wpid6_test_child(i, pgid);
      }
   }

   cld[6] = fork();
   DEVSHELL_CMD_ASSERT(cld[6] >= 0);

   if (!cld[6])
      wpid6_test_active_child(6, cld[0], cld[3]);

   printf(STR_PARENT "Wait for children to change their pgid\n");
   usleep(100 * 1000);

   printf(STR_PARENT "Wait on the 1st process group\n");
   while ((pid = call_waitpid(-cld[0], &wstatus, cld, ARRAY_SIZE(cld))) > 0) {
      if (pid == cld[0] || pid == cld[1] || pid == cld[2])
         g1_killed++;
   }

   DEVSHELL_CMD_ASSERT(g1_killed == 3);

   printf(STR_PARENT "Wait on the 2st process group\n");
   while ((pid = call_waitpid(-cld[3], &wstatus, cld, ARRAY_SIZE(cld))) > 0) {
      if (pid == cld[3] || pid == cld[4] || pid == cld[5])
         g2_killed++;
   }

   DEVSHELL_CMD_ASSERT(g2_killed == 3);

   printf(STR_PARENT "Wait on any other child\n");
   pid = call_waitpid(-1, &wstatus, cld, ARRAY_SIZE(cld));
   DEVSHELL_CMD_ASSERT(pid == cld[6]);
   return 0;
}
