#include "startappprocess.h"
#include <QCoreApplication>
#include <QTextStream>
#include <QRegularExpression>
#include <QProgressDialog>
#include <QStandardPaths>
#include <QThread>
#include <QDebug>

StartAppProcess::StartAppProcess(QObject* parent)
    : AdbProcess(parent) {

}

void StartAppProcess::StartApp(const QString& appName, const QString& subProcessName, const QString& compiler, 
    const QString& arch, bool interceptMode, QProgressDialog* dialog) {
    startResult_ = false;
    interceptMode_ = interceptMode;
    errorStr_ = QString();
    auto execPath = GetExecutablePath();
    QStringList arguments;
    { // push remote folder to /data/local/tmp
        dialog->setLabelText("Pushing libloli.so to device.");
        arguments.clear();
        arguments << "push" << "remote/" + compiler + "/" + arch + "/libloli.so" << "/data/local/tmp";
        QProcess process;
        process.setWorkingDirectory(QCoreApplication::applicationDirPath());
        process.setProgram(execPath);
        AdbProcess::SetArguments(&process, arguments);
        if (!StartProcess(&process, "adb push remote/libloli.so /data/local/tmp")) {
            return;
        }
        dialog->setValue(dialog->value() + 1);
    }
    { // check if device is rooted or not.
        isRootDevice_ = false;
        arguments.clear();
        arguments << "shell" << "su";
        QProcess process;
        process.setProgram(execPath);
        AdbProcess::SetArguments(&process, arguments);
        process.start();
        if (process.waitForStarted()) {
            if (!process.waitForFinished(3000)) {
                isRootDevice_ = QString(process.readAll()).size() == 0;
                process.close();
            }
        }
    }
    { // push remote folder to /data/local/tmp
        dialog->setLabelText("Pushing loli.conf to device.");
        arguments.clear();
        arguments << "push" << "loli2.conf" << "/data/local/tmp";
        QProcess process;
        process.setWorkingDirectory(QStandardPaths::standardLocations(QStandardPaths::AppDataLocation).first());
        process.setProgram(execPath);
        AdbProcess::SetArguments(&process, arguments);
        if (!StartProcess(&process, "adb push loli2.conf /data/local/tmp")) {
            return;
        }
        dialog->setValue(dialog->value() + 1);
    }
    if (!interceptMode) { // set app as debugable for next launch
        dialog->setLabelText("Marking apk debugable for next launch.");
        arguments.clear();
        arguments << "shell" << "am" << "set-debug-app" << "-w" << appName;
        QProcess process;
        process.setProgram(execPath);
        AdbProcess::SetArguments(&process, arguments);
        if (!StartProcess(&process, "adb shell am set-debug-app -w com.company.app")) {
            return;
        }
        dialog->setValue(dialog->value() + 1);
    }
    if (!interceptMode) { // launch the app
        dialog->setLabelText("Launching apk.");
        arguments.clear();
        arguments << "shell" << "monkey -p" << appName << "-c android.intent.category.LAUNCHER 1";
        QProcess process;
        process.setProgram(execPath);
        AdbProcess::SetArguments(&process, arguments);
        if (!StartProcess(&process, "adb shell monkey -p com.company.app -c android.intent.category.LAUNCHER 1")) {
            return;
        }
        dialog->setValue(dialog->value() + 1);
    }
    unsigned int pid = 0;
    { // pid of
        QThread::sleep(2);
        dialog->setLabelText("Gettting pid.");
        arguments.clear();
        // if need attch subProcess
        // https://stackoverflow.com/questions/15608876/find-out-the-running-process-id-by-package-name
        if (subProcessName.isNull() || subProcessName.isEmpty()) {
            arguments << "shell" << 
                "for p in /proc/[0-9]*; do [[ $(<$p/cmdline) = " + appName + " ]] && echo ${p##*/}; done";
        } else {
            arguments << "shell" << 
                "for p in /proc/[0-9]*; do [[ $(<$p/cmdline) = " + appName + ":" + 
                subProcessName + " ]] && echo ${p##*/}; done";
        }
        QProcess process;
        process.setProgram(execPath);
        AdbProcess::SetArguments(&process, arguments);
        if (!StartProcess(&process, "looking for pid")) {
            return;
        }
        pid = process.readAll().trimmed().toUInt();
//        qDebug() << pid;
        dialog->setValue(dialog->value() + 1);
    }
    { // adb forward
        dialog->setLabelText("Forwadring tcp port.");
        arguments.clear();
        arguments << "forward" << "tcp:8700" << ("jdwp:" + QString::number(pid));
        QProcess process;
        process.setProgram(execPath);
        AdbProcess::SetArguments(&process, arguments);
        if (!StartProcess(&process, "adb forward tcp:8700 jdwp:xxxx")) {
            return;
        }
        dialog->setValue(dialog->value() + 1);
    }
    // python jdwp-shellifier.py
    errorStr_ = "python jdwp-shellifier.py";
    dialog->setLabelText("Injecting libloli.so to target application.");
    dialog->setCancelButtonText("Cancel");
    dialog->raise();
    arguments.clear();
    arguments << "jdwp-shellifier.py" << "--target" << "127.0.0.1" << "--port" << "8700" << 
        "--break-on" << "android.app.Activity.onResume" << "--loadlib" << "libloli.so";
    process_->setWorkingDirectory(QCoreApplication::applicationDirPath());
    process_->setProgram(pythonPath_);
    AdbProcess::SetArguments(process_, arguments);
    ExecuteAsync();
}

bool StartAppProcess::GetSMapsByRunAs(const QString& appName, const QString& appPid) {
    auto execPath = GetExecutablePath();
    QStringList arguments;
    QProcess process;
    process.setProgram(execPath);
    if (isRootDevice_) {
        arguments << "shell" << "su" << "-c" << "\"cat" << "/proc/" + appPid + 
            "/smaps" << ">" << "/data/local/tmp/smaps.txt\"";
    } else {
        arguments << "shell" << "run-as" << appName << "cat" << "/proc/" + appPid + 
            "/smaps" << ">" << "/data/local/tmp/smaps.txt";
    }
    AdbProcess::SetArguments(&process, arguments);
    process.start();
    if (!process.waitForStarted()) {
        return false;
    }
    if (!process.waitForFinished()) {
        return false;
    }
    if (QString(process.readAll()).size() > 0) {
        return false;
    }
    return true;
}

bool StartAppProcess::StartProcess(QProcess* process, const QString& message) {
    process->start();
    if (!process->waitForStarted()) {
        errorStr_ = "erro starting: " + message;
        emit ProcessErrorOccurred();
        return false;
    }
    if (!process->waitForFinished()) {
        errorStr_ = "erro finishing: " + message;
        emit ProcessErrorOccurred();
        return false;
    }
    return true;
}

void StartAppProcess::OnProcessFinihed() {
    QString retStr = process_->readAll();
    process_->close();
    QTextStream stream(&retStr);
    QString line;
    while(stream.readLineInto(&line)) {
        if (line.contains("Command successfully executed"))
        {
            startResult_ = true;
            break;
        }
    }
}

void StartAppProcess::OnProcessErrorOccurred() {
    errorStr_ = process_->readAll();
}
