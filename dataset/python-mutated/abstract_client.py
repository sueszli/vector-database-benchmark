"""
Module containing abstract class about hdfs clients.
"""
import abc
import luigi.target

class HdfsFileSystem(luigi.target.FileSystem, metaclass=abc.ABCMeta):
    """
    This client uses Apache 2.x syntax for file system commands, which also matched CDH4.
    """

    def rename(self, path, dest):
        if False:
            i = 10
            return i + 15
        '\n        Rename or move a file.\n\n        In hdfs land, "mv" is often called rename. So we add an alias for\n        ``move()`` called ``rename()``. This is also to keep backward\n        compatibility since ``move()`` became standardized in luigi\'s\n        filesystem interface.\n        '
        return self.move(path, dest)

    def rename_dont_move(self, path, dest):
        if False:
            i = 10
            return i + 15
        '\n        Override this method with an implementation that uses rename2,\n        which is a rename operation that never moves.\n\n        rename2 -\n        https://github.com/apache/hadoop/blob/ae91b13/hadoop-hdfs-project/hadoop-hdfs/src/main/java/org/apache/hadoop/hdfs/protocol/ClientProtocol.java\n        (lines 483-523)\n        '
        return super(HdfsFileSystem, self).rename_dont_move(path, dest)

    @abc.abstractmethod
    def remove(self, path, recursive=True, skip_trash=False):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abc.abstractmethod
    def chmod(self, path, permissions, recursive=False):
        if False:
            return 10
        pass

    @abc.abstractmethod
    def chown(self, path, owner, group, recursive=False):
        if False:
            print('Hello World!')
        pass

    @abc.abstractmethod
    def count(self, path):
        if False:
            i = 10
            return i + 15
        '\n        Count contents in a directory\n        '
        pass

    @abc.abstractmethod
    def copy(self, path, destination):
        if False:
            return 10
        pass

    @abc.abstractmethod
    def put(self, local_path, destination):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abc.abstractmethod
    def get(self, path, local_destination):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abc.abstractmethod
    def mkdir(self, path, parents=True, raise_if_exists=False):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abc.abstractmethod
    def listdir(self, path, ignore_directories=False, ignore_files=False, include_size=False, include_type=False, include_time=False, recursive=False):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abc.abstractmethod
    def touchz(self, path):
        if False:
            i = 10
            return i + 15
        pass