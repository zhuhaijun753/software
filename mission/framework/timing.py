from mission.framework.combinators import MasterConcurrent
from mission.framework.helpers import call_if_function
from mission.framework.task import Task


class Timer(Task):
    """ A task that finishes after a set amount of time.

        Args:
            seconds: The amount of seconds to be waited before finishing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seconds = 0

    def on_run(self, seconds, *args, **kwargs):
        """
        Args:
            seconds: The amount of seconds to be waited before finishing.
        """
        self.seconds = call_if_function(seconds)
        if (self.this_run_time - self.first_run_time) >= self.seconds:
            self.finish()

class GuardedTimer(Task):
    """ A task that finishes after a set amount of time.

        Args:
            seconds: The amount of seconds to be waited before finishing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seconds = 0
        self.state = 'run'

    def on_run(self, seconds, task, restore, *args, **kwargs):
        """
        Args:
            seconds: The amount of seconds to be waited before finishing.
        """
        if self.state == 'run':
            self.seconds = call_if_function(seconds)
            task()
            if task.finished:
                self.finish()

            if (self.this_run_time - self.first_run_time) >= self.seconds:
                self.state = 'restore'
                return
            
        if self.state == 'restore':
            restore()
            if restore.finished: self.finish()

class Timeout(Task):
    """
    Try doing a task for a certain amount of time

    We are successful if the task completes in time and is successful.
    """
    def on_first_run(self, task, time, *args, **kwargs):
        self.success = False
        self.task = task
        self.timer = Timer(time)
        self.timed_out = False

    def on_run(self, *args, **kwargs):
        self.task()
        self.timer()
        if self.task.finished:
            if hasattr(self.task, 'success'):
                self.success = self.task.success
            else:
                self.success = True
            self.finish()
        elif self.timer.finished:
            self.timed_out = True
            self.finish()

Timed = lambda task, seconds: MasterConcurrent(Timer(seconds), task)
