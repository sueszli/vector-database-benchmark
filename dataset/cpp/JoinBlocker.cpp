/*
	This file is part of duckOS.

	duckOS is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	duckOS is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with duckOS.  If not, see <https://www.gnu.org/licenses/>.

	Copyright (c) Byteduck 2016-2021. All rights reserved.
*/

#include "JoinBlocker.h"
#include "Thread.h"

JoinBlocker::JoinBlocker(kstd::Arc<Thread> thread, kstd::Arc<Thread> wait_for) {
	_thread = thread;
	_wait_thread = wait_for;
}

bool JoinBlocker::is_ready() {
	return _wait_thread->state() == Thread::ZOMBIE || _wait_thread->state() == Thread::DEAD;
}

kstd::Arc<Thread> JoinBlocker::waited_thread() {
	return _wait_thread;
}