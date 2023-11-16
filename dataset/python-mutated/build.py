"""
build.py: build a Mininet VM

Basic idea:

    prepare
    -> create base install image if it's missing
        - download iso if it's missing
        - install from iso onto image

    build
    -> create cow disk for new VM, based on base image
    -> boot it in qemu/kvm with text /serial console
    -> install Mininet

    test
    -> sudo mn --test pingall
    -> make test

    release
    -> shut down VM
    -> shrink-wrap VM
    -> upload to storage

"""
import os
from os import stat, path
from stat import ST_MODE, ST_SIZE
from os.path import abspath
from sys import exit, stdout, argv, modules
import sys
import re
from glob import glob
from subprocess import check_output, call, Popen
from tempfile import mkdtemp, NamedTemporaryFile
from time import time, strftime, localtime
import argparse
from distutils.spawn import find_executable
import inspect
from traceback import print_exc
pexpect = None
TIMEOUT = 600
LogToConsole = False
SaveQCOW2 = False
NoKVM = False
Branch = None
Zip = False
Forward = []
Chown = ''
VMImageDir = os.environ['HOME'] + '/vm-images'
Prompt = '\\$ '

def serverURL(version, arch):
    if False:
        for i in range(10):
            print('nop')
    'Return .iso URL for Ubuntu version and arch'
    server = 'http://cdimage.ubuntu.com/ubuntu/releases/%s/release/'
    iso = 'ubuntu-%s-server-%s.iso'
    return (server + iso) % (version, version, arch)

def legacyURL(version, arch):
    if False:
        print('Hello World!')
    'Return .iso URL for Ubuntu version'
    server = 'http://cdimage.ubuntu.com/ubuntu-legacy-server/releases/%s/release/'
    iso = 'ubuntu-%s-legacy-server-%s.iso'
    return (server + iso) % (version, version, arch)
isoURLs = {'precise64server': 'http://mirrors.kernel.org/ubuntu-releases/12.04/ubuntu-12.04.5-server-amd64.iso', 'trusty32server': 'http://mirrors.kernel.org/ubuntu-releases/14.04/ubuntu-14.04.4-server-i386.iso', 'trusty64server': 'http://mirrors.kernel.org/ubuntu-releases/14.04/ubuntu-14.04.4-server-amd64.iso', 'xenial32server': 'http://mirrors.kernel.org/ubuntu-releases/16.04/ubuntu-16.04.6-server-i386.iso', 'xenial64server': 'http://mirrors.kernel.org/ubuntu-releases/16.04/ubuntu-16.04.7-server-amd64.iso', 'bionic64server': serverURL('18.04.5', 'amd64'), 'focal64server': legacyURL('20.04.1', 'amd64')}

def OSVersion(flavor):
    if False:
        while True:
            i = 10
    'Return full OS version string for build flavor'
    urlbase = path.basename(isoURLs.get(flavor, 'unknown'))
    return path.splitext(urlbase)[0]

def OVFOSNameID(flavor):
    if False:
        i = 10
        return i + 15
    'Return OVF-specified ( OS Name, ID ) for flavor'
    version = OSVersion(flavor)
    arch = archFor(flavor)
    if 'ubuntu' in version:
        map = {'i386': ('Ubuntu', 93), 'x86_64': ('Ubuntu 64-bit', 94)}
    else:
        map = {'i386': ('Linux', 36), 'x86_64': ('Linux 64-bit', 101)}
    (osname, osid) = map[arch]
    return (osname, osid)
LogStartTime = time()
LogFile = None

def log(*args, **kwargs):
    if False:
        return 10
    'Simple log function: log( message along with local and elapsed time\n       cr: False/0 for no CR'
    cr = kwargs.get('cr', True)
    elapsed = time() - LogStartTime
    clocktime = strftime('%H:%M:%S', localtime())
    msg = ' '.join((str(arg) for arg in args))
    output = '%s [ %.3f ] %s' % (clocktime, elapsed, msg)
    if cr:
        print(output)
    else:
        print(output)
    if LogFile:
        if cr:
            output += '\n'
        LogFile.write(output)
        LogFile.flush()

def run(cmd, **kwargs):
    if False:
        i = 10
        return i + 15
    'Convenient interface to check_output'
    log('+', cmd)
    cmd = cmd.split()
    arg0 = cmd[0]
    if not find_executable(arg0):
        raise Exception('Cannot find executable "%s";' % arg0 + 'you might try %s --depend' % argv[0])
    return check_output(cmd, **kwargs)

def srun(cmd, **kwargs):
    if False:
        return 10
    'Run + sudo'
    return run('sudo ' + cmd, **kwargs)

def depend():
    if False:
        while True:
            i = 10
    'Install package dependencies'
    log('* Installing package dependencies')
    run('sudo apt-get -qy update')
    run('sudo apt-get -qy install kvmtool cloud-utils genisoimage qemu-kvm qemu-utils e2fsprogs curl python-setuptools mtools zip')
    run('sudo easy_install pexpect')

def popen(cmd):
    if False:
        print('Hello World!')
    'Convenient interface to popen'
    log(cmd)
    cmd = cmd.split()
    return Popen(cmd)

def remove(fname):
    if False:
        i = 10
        return i + 15
    'Remove a file, ignoring errors'
    try:
        os.remove(fname)
    except OSError:
        pass

def findiso(flavor):
    if False:
        for i in range(10):
            print('nop')
    "Find iso, fetching it if it's not there already"
    url = isoURLs[flavor]
    name = path.basename(url)
    iso = path.join(VMImageDir, name)
    if not path.exists(iso) or stat(iso)[ST_MODE] & 511 != 292:
        log('* Retrieving', url)
        run('curl -L -C - -o %s %s' % (iso, url))
        result = run('file ' + iso)
        if 'ISO' not in result and 'boot' not in result:
            os.remove(iso)
            raise Exception('findiso: could not download iso from ' + url)
        log('* Write-protecting iso', iso)
        os.chmod(iso, 292)
    log('* Using iso', iso)
    return iso

def attachNBD(cow, flags=''):
    if False:
        i = 10
        return i + 15
    'Attempt to attach a COW disk image and return its nbd device\n        flags: additional flags for qemu-nbd (e.g. -r for readonly)'
    cow = abspath(cow)
    log('* Checking for unused /dev/nbdX device ')
    for i in range(1, 63):
        entry = 'nbd%d' % i
        nbd = '/dev/' + entry
        if call(['pgrep', '-f', entry]) == 0:
            continue
        srun('modprobe nbd max-part=64')
        srun('qemu-nbd %s -c %s %s' % (flags, nbd, cow))
        print()
        return nbd
    raise Exception('Error: could not find unused /dev/nbdX device')

def detachNBD(nbd):
    if False:
        for i in range(10):
            print('nop')
    'Detach an nbd device'
    srun('qemu-nbd -d ' + nbd)

def extractKernel(image, flavor, imageDir=VMImageDir):
    if False:
        while True:
            i = 10
    'Extract kernel and initrd from base image'
    kernel = path.join(imageDir, flavor + '-vmlinuz')
    initrd = path.join(imageDir, flavor + '-initrd')
    log('* Extracting kernel to', kernel)
    nbd = attachNBD(image, flags='-r')
    try:
        print(srun('partx ' + nbd))
    except:
        log('Warning - partx failed with error')
    part = nbd + 'p1'
    partitions = srun('fdisk -l ' + nbd)
    for line in partitions.split('\n'):
        line = line.strip()
        if line.endswith('Linux'):
            part = line.split()[0]
            break
    partnum = int(part.split('p')[-1])
    if path.exists(kernel) and stat(image)[ST_MODE] & 511 == 292:
        detachNBD(nbd)
        return (kernel, initrd, partnum)
    mnt = mkdtemp()
    srun('mount -o ro,noload %s %s' % (part, mnt))
    kernsrc = glob('%s/boot/vmlinuz*generic' % mnt)[0]
    initrdsrc = glob('%s/boot/initrd*generic' % mnt)[0]
    srun('cp %s %s' % (initrdsrc, initrd))
    srun('chmod 0444 ' + initrd)
    srun('cp %s %s' % (kernsrc, kernel))
    srun('chmod 0444 ' + kernel)
    srun('umount ' + mnt)
    run('rmdir ' + mnt)
    detachNBD(nbd)
    return (kernel, initrd, partnum)

def findBaseImage(flavor, size='8G'):
    if False:
        print('Hello World!')
    'Return base VM image and kernel, creating them if needed'
    image = path.join(VMImageDir, flavor + '-base.qcow2')
    if path.exists(image):
        perms = stat(image)[ST_MODE] & 511
        if perms != 292:
            raise Exception('Error - base image %s is writable.' % image + ' Are multiple builds running? if not, remove %s and try again.' % image)
    else:
        run('mkdir -p %s' % VMImageDir)
        iso = findiso(flavor)
        log('* Creating image file', image)
        run('qemu-img create -f qcow2 %s %s' % (image, size))
        installUbuntu(iso, image)
        log('* Write-protecting image', image)
        os.chmod(image, 292)
    (kernel, initrd, partnum) = extractKernel(image, flavor)
    log('* Using base image', image, 'and kernel', kernel, 'and partition #', partnum)
    return (image, kernel, initrd, partnum)
KickstartText = '\n#Generated by Kickstart Configurator\n#platform=x86\n\n#System language\nlang en_US\n#Language modules to install\nlangsupport en_US\n#System keyboard\nkeyboard us\n#System mouse\nmouse\n#System timezone\ntimezone America/Los_Angeles\n#Root password\nrootpw --disabled\n#Initial user\nuser mininet --fullname "mininet" --password "mininet"\n#Use text mode install\ntext\n#Install OS instead of upgrade\ninstall\n#Use CDROM installation media\ncdrom\n#System bootloader configuration\nbootloader --location=mbr\n#Clear the Master Boot Record\nzerombr yes\n#Partition clearing information\nclearpart --all --initlabel\n#Automatic partitioning\nautopart\n#System authorization information\nauth  --useshadow  --enablemd5\n#Firewall configuration\nfirewall --disabled\n#Do not configure the X Window System\nskipx\n'
PreseedText = '\n\nd-i mirror/http/directory string /ubuntu\nd-i mirror/http/proxy string\nd-i partman/confirm_write_new_label boolean true\nd-i partman/choose_partition select finish\nd-i partman/confirm boolean true\nd-i partman/confirm_nooverwrite boolean true\nd-i user-setup/allow-password-weak boolean true\nd-i finish-install/reboot_in_progress note\nd-i debian-installer/exit/poweroff boolean true\n'

def makeKickstartFloppy():
    if False:
        for i in range(10):
            print('nop')
    'Create and return kickstart floppy, kickstart, preseed'
    kickstart = 'ks.cfg'
    with open(kickstart, 'w') as f:
        f.write(KickstartText)
    preseed = 'ks.preseed'
    with open(preseed, 'w') as f:
        f.write(PreseedText)
    floppy = 'ksfloppy.img'
    run('qemu-img create %s 1440k' % floppy)
    run('mkfs -t msdos ' + floppy)
    run('mcopy -i %s %s ::/' % (floppy, kickstart))
    run('mcopy -i %s %s ::/' % (floppy, preseed))
    return (floppy, kickstart, preseed)

def archFor(filepath):
    if False:
        i = 10
        return i + 15
    'Guess architecture for file path'
    name = path.basename(filepath)
    if 'amd64' in name or 'x86_64' in name:
        arch = 'x86_64'
    elif 'i386' in name or '32' in name or 'x86' in name:
        arch = 'i386'
    elif '64' in name:
        arch = 'x86_64'
    else:
        log("Error: can't discern CPU for name", name)
        exit(1)
    return arch

def installUbuntu(iso, image, logfilename='install.log', memory=1024):
    if False:
        while True:
            i = 10
    'Install Ubuntu from iso onto image'
    kvm = 'qemu-system-' + archFor(iso)
    (floppy, kickstart, preseed) = makeKickstartFloppy()
    mnt = mkdtemp()
    srun('mount %s %s' % (iso, mnt))
    for kdir in ('install', 'casper'):
        kernel = path.join(mnt, kdir, 'vmlinuz')
        if not path.exists(kernel):
            kernel = ''
        for initrd in ('initrd.gz', 'initrd'):
            initrd = path.join(mnt, kdir, initrd)
            if path.exists(initrd):
                break
            else:
                initrd = ''
        if kernel and initrd:
            break
    if not kernel or not initrd:
        raise Exception('unable to locate kernel and initrd in iso image')
    if NoKVM:
        accel = 'tcg'
    else:
        accel = 'kvm'
        try:
            run('kvm-ok')
        except:
            raise Exception('kvm-ok failed; try using --nokvm')
    cmd = ['sudo', kvm, '-machine', 'accel=%s' % accel, '-nographic', '-netdev', 'user,id=mnbuild', '-device', 'virtio-net,netdev=mnbuild', '-m', str(memory), '-k', 'en-us', '-fda', floppy, '-drive', 'file=%s,if=virtio' % image, '-cdrom', iso, '-kernel', kernel, '-initrd', initrd, '-append', ' ks=floppy:/' + kickstart + ' preseed/file=floppy://' + preseed + ' net.ifnames=0' + ' console=ttyS0']
    ubuntuStart = time()
    log('* INSTALLING UBUNTU FROM', iso, 'ONTO', image)
    log(' '.join(cmd))
    log('* logging to', abspath(logfilename))
    params = {}
    if not LogToConsole:
        logfile = open(logfilename, 'w')
        params = {'stdout': logfile, 'stderr': logfile}
    vm = Popen(cmd, **params)
    log('* Waiting for installation to complete')
    vm.wait()
    if not LogToConsole:
        logfile.close()
    elapsed = time() - ubuntuStart
    srun('ls -l ' + mnt)
    srun('umount ' + mnt)
    run('rmdir ' + mnt)
    if vm.returncode != 0:
        raise Exception('Ubuntu installation returned error %d' % vm.returncode)
    log('* UBUNTU INSTALLATION COMPLETED FOR', image)
    log('* Ubuntu installation completed in %.2f seconds' % elapsed)

def boot(cow, kernel, initrd, logfile, memory=1024, cpuCores=1, partnum=1):
    if False:
        print('Hello World!')
    'Boot qemu/kvm with a COW disk and local/user data store\n       cow: COW disk path\n       kernel: kernel path\n       logfile: log file for pexpect object\n       memory: memory size in MB\n       cpuCores: number of CPU cores to use\n       returns: pexpect object to qemu process'
    global pexpect
    if not pexpect:
        import pexpect

    class Spawn(pexpect.spawn):
        """Subprocess is sudo, so we have to sudo kill it"""

        def close(self, force=False):
            if False:
                print('Hello World!')
            srun('kill %d' % self.pid)
    arch = archFor(kernel)
    log('* Detected kernel architecture', arch)
    if NoKVM:
        accel = 'tcg'
    else:
        accel = 'kvm'
    cmd = ['sudo', 'qemu-system-' + arch, '-machine accel=%s' % accel, '-nographic', '-netdev user,id=mnbuild', '-device virtio-net,netdev=mnbuild', '-m %s' % memory, '-k en-us', '-kernel', kernel, '-initrd', initrd, '-drive file=%s,if=virtio' % cow, '-append "root=/dev/vda%d init=/sbin/init net.ifnames=0 console=ttyS0" ' % partnum]
    log(cmd)
    if Forward:
        cmd += sum([['-redir', f] for f in Forward], [])
    if cpuCores > 1:
        cmd += ['-smp cores=%s' % cpuCores]
    cmd = ' '.join(cmd)
    log('* BOOTING VM FROM', cow)
    log(cmd)
    vm = Spawn(cmd, timeout=TIMEOUT, logfile=logfile)
    return vm

def login(vm, user='mininet', password='mininet'):
    if False:
        return 10
    'Log in to vm (pexpect object)'
    log('* Waiting for login prompt')
    vm.expect('login: ')
    log('* Logging in')
    vm.sendline(user)
    log('* Waiting for password prompt')
    vm.expect('Password: ')
    log('* Sending password')
    vm.sendline(password)
    log('* Waiting for login...')

def removeNtpd(vm, prompt=Prompt, ntpPackage='ntp'):
    if False:
        return 10
    'Remove ntpd and set clock immediately'
    log('* Removing ntpd')
    vm.sendline('sudo -n apt-get -qy remove ' + ntpPackage)
    vm.expect(prompt)
    vm.sendline('sudo -n pkill ntpd')
    vm.expect(prompt)
    log('* Getting seconds since epoch from this server')
    seconds = int(run('date +%s'))
    log('* Setting VM clock')
    vm.sendline('sudo -n date -s @%d' % seconds)

def sanityTest(vm):
    if False:
        i = 10
        return i + 15
    'Run Mininet sanity test (pingall) in vm'
    vm.sendline('sudo -n mn --test pingall')
    if vm.expect([' 0% dropped', pexpect.TIMEOUT], timeout=45) == 0:
        log('* Sanity check OK')
    else:
        log('* Sanity check FAILED')
        log('* Sanity check output:')
        log(vm.before)

def coreTest(vm, prompt=Prompt):
    if False:
        return 10
    'Run core tests (make test) in VM'
    log('* Making sure cgroups are mounted')
    vm.sendline('sudo -n service cgroup-lite restart')
    vm.expect(prompt)
    vm.sendline('sudo -n cgroups-mount')
    vm.expect(prompt)
    log('* Running make test')
    vm.sendline('cd ~/mininet; sudo make test')
    for test in range(0, 2):
        if vm.expect(['OK.*\r\n', 'FAILED.*\r\n', pexpect.TIMEOUT], timeout=180) == 0:
            log('* Test', test, 'OK')
        else:
            log('* Test', test, 'FAILED')
            log('* Test', test, 'output:')
            log(vm.before)

def installPexpect(vm, prompt=Prompt):
    if False:
        i = 10
        return i + 15
    'install pexpect'
    vm.sendline('sudo -n apt-get -qy install python-pexpect')
    vm.expect(prompt)

def noneTest(vm, prompt=Prompt):
    if False:
        print('Hello World!')
    'This test does nothing'
    installPexpect(vm, prompt)
    vm.sendline('echo')

def examplesquickTest(vm, prompt=Prompt):
    if False:
        for i in range(10):
            print('nop')
    'Quick test of mininet examples'
    installPexpect(vm, prompt)
    vm.sendline('sudo -n python ~/mininet/examples/test/runner.py -v -quick')

def examplesfullTest(vm, prompt=Prompt):
    if False:
        for i in range(10):
            print('nop')
    'Full (slow) test of mininet examples'
    installPexpect(vm, prompt)
    vm.sendline('sudo -n python ~/mininet/examples/test/runner.py -v')

def walkthroughTest(vm, prompt=Prompt):
    if False:
        print('Hello World!')
    'Test mininet walkthrough'
    installPexpect(vm, prompt)
    vm.sendline('sudo -n python ~/mininet/mininet/test/test_walkthrough.py -v')

def useTest(vm, prompt=Prompt):
    if False:
        i = 10
        return i + 15
    'Use VM interactively - exit by pressing control-]'
    old = vm.logfile
    if old == stdout:
        log('* Temporarily disabling logging to stdout')
        vm.logfile = None
    log('* Switching to interactive use - press control-] to exit')
    vm.interact()
    if old == stdout:
        log('* Restoring logging to stdout')
        vm.logfile = stdout
runTest = useTest

def checkOutBranch(vm, branch, prompt=Prompt):
    if False:
        i = 10
        return i + 15
    vm.sendline('cd ~/mininet; git fetch origin ' + branch + '; git checkout ' + branch + '; git pull --rebase origin ' + branch)
    vm.expect(prompt)
    vm.sendline('util/install.sh -n')

def interact(vm, tests, pre='', post='', prompt=Prompt, clean=True):
    if False:
        print('Hello World!')
    'Interact with vm, which is a pexpect object'
    login(vm)
    log('* Waiting for login...')
    vm.expect(prompt)
    log('* Sending hostname command')
    vm.sendline('hostname')
    log('* Waiting for output')
    vm.expect(prompt)
    log('* Fetching Mininet VM install script')
    branch = Branch if Branch else 'master'
    vm.sendline('wget https://raw.github.com/mininet/mininet/%s/util/vm/install-mininet-vm.sh' % branch)
    vm.expect(prompt)
    log('* Running VM install script')
    installcmd = 'bash -v install-mininet-vm.sh'
    if Branch:
        installcmd += ' ' + Branch
    vm.sendline(installcmd)
    vm.expect('password for mininet: ')
    vm.sendline('mininet')
    log('* Waiting for script to complete... ')
    timeout = 5200 if NoKVM else 1800
    vm.expect('Done preparing Mininet', timeout=timeout)
    log('* Completed successfully')
    vm.expect(prompt)
    version = getMininetVersion(vm)
    vm.expect(prompt)
    log('* Mininet version: ', version)
    log('* Testing Mininet')
    runTests(vm, tests=tests, pre=pre, post=post)
    log('* Disabling serial console')
    vm.sendline("sudo sed -i -e 's/^GRUB_TERMINAL=serial/#GRUB_TERMINAL=serial/' /etc/default/grub; sudo update-grub")
    vm.expect(prompt)
    if clean:
        log('* Cleaning vm')
        vm.sendline('~/mininet/util/install.sh -d')
    vm.expect(prompt)
    log('* Shutting down')
    vm.sendline('sync; sudo shutdown -h now')
    log('* Waiting for EOF/shutdown')
    vm.read()
    log('* Interaction complete')
    return version

def cleanup():
    if False:
        print('Hello World!')
    'Clean up leftover qemu-nbd processes and other junk'
    call(['sudo', 'pkill', '-9', 'qemu-nbd'])

def convert(cow, basename):
    if False:
        i = 10
        return i + 15
    'Convert a qcow2 disk to a vmdk and put it a new directory\n       basename: base name for output vmdk file'
    vmdk = basename + '.vmdk'
    log('* Converting qcow2 to vmdk')
    run('qemu-img convert -f qcow2 -O vmdk %s %s' % (cow, vmdk))
    return vmdk
OVFTemplate = '<?xml version="1.0"?>\n<Envelope ovf:version="1.0" xml:lang="en-US"\n    xmlns="http://schemas.dmtf.org/ovf/envelope/1"\n    xmlns:ovf="http://schemas.dmtf.org/ovf/envelope/1"\n    xmlns:rasd="http://schemas.dmtf.org/wbem/wscim/1/cim-schema/2/CIM_ResourceAllocationSettingData"\n    xmlns:vssd="http://schemas.dmtf.org/wbem/wscim/1/cim-schema/2/CIM_VirtualSystemSettingData"\n    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n<References>\n<File ovf:href="%(diskname)s" ovf:id="file1" ovf:size="%(filesize)d"/>\n</References>\n<DiskSection>\n<Info>Virtual disk information</Info>\n<Disk ovf:capacity="%(disksize)d" ovf:capacityAllocationUnits="byte"\n    ovf:diskId="vmdisk1" ovf:fileRef="file1"\n    ovf:format="http://www.vmware.com/interfaces/specifications/vmdk.html#streamOptimized"/>\n</DiskSection>\n<NetworkSection>\n<Info>The list of logical networks</Info>\n<Network ovf:name="nat">\n<Description>The nat  network</Description>\n</Network>\n</NetworkSection>\n<VirtualSystem ovf:id="%(vmname)s">\n<Info>%(vminfo)s (%(name)s)</Info>\n<Name>%(vmname)s</Name>\n<OperatingSystemSection ovf:id="%(osid)d">\n<Info>The kind of installed guest operating system</Info>\n<Description>%(osname)s</Description>\n</OperatingSystemSection>\n<VirtualHardwareSection>\n<Info>Virtual hardware requirements</Info>\n<Item>\n<rasd:AllocationUnits>hertz * 10^6</rasd:AllocationUnits>\n<rasd:Description>Number of Virtual CPUs</rasd:Description>\n<rasd:ElementName>%(cpus)s virtual CPU(s)</rasd:ElementName>\n<rasd:InstanceID>1</rasd:InstanceID>\n<rasd:ResourceType>3</rasd:ResourceType>\n<rasd:VirtualQuantity>%(cpus)s</rasd:VirtualQuantity>\n</Item>\n<Item>\n<rasd:AllocationUnits>byte * 2^20</rasd:AllocationUnits>\n<rasd:Description>Memory Size</rasd:Description>\n<rasd:ElementName>%(mem)dMB of memory</rasd:ElementName>\n<rasd:InstanceID>2</rasd:InstanceID>\n<rasd:ResourceType>4</rasd:ResourceType>\n<rasd:VirtualQuantity>%(mem)d</rasd:VirtualQuantity>\n</Item>\n<Item>\n<rasd:Address>0</rasd:Address>\n<rasd:Caption>scsiController0</rasd:Caption>\n<rasd:Description>SCSI Controller</rasd:Description>\n<rasd:ElementName>scsiController0</rasd:ElementName>\n<rasd:InstanceID>4</rasd:InstanceID>\n<rasd:ResourceSubType>lsilogic</rasd:ResourceSubType>\n<rasd:ResourceType>6</rasd:ResourceType>\n</Item>\n<Item>\n<rasd:AddressOnParent>0</rasd:AddressOnParent>\n<rasd:ElementName>disk1</rasd:ElementName>\n<rasd:HostResource>ovf:/disk/vmdisk1</rasd:HostResource>\n<rasd:InstanceID>11</rasd:InstanceID>\n<rasd:Parent>4</rasd:Parent>\n<rasd:ResourceType>17</rasd:ResourceType>\n</Item>\n<Item>\n<rasd:AddressOnParent>2</rasd:AddressOnParent>\n<rasd:AutomaticAllocation>true</rasd:AutomaticAllocation>\n<rasd:Connection>nat</rasd:Connection>\n<rasd:Description>E1000 ethernet adapter on nat</rasd:Description>\n<rasd:ElementName>ethernet0</rasd:ElementName>\n<rasd:InstanceID>12</rasd:InstanceID>\n<rasd:ResourceSubType>E1000</rasd:ResourceSubType>\n<rasd:ResourceType>10</rasd:ResourceType>\n</Item>\n<Item>\n<rasd:Address>0</rasd:Address>\n<rasd:Caption>usb</rasd:Caption>\n<rasd:Description>USB Controller</rasd:Description>\n<rasd:ElementName>usb</rasd:ElementName>\n<rasd:InstanceID>9</rasd:InstanceID>\n<rasd:ResourceType>23</rasd:ResourceType>\n</Item>\n</VirtualHardwareSection>\n</VirtualSystem>\n</Envelope>\n'

def generateOVF(name, osname, osid, diskname, disksize, mem=1024, cpus=1, vmname='Mininet-VM', vminfo='A Mininet Virtual Machine'):
    if False:
        print('Hello World!')
    'Generate (and return) OVF file "name.ovf"\n       name: root name of OVF file to generate\n       osname: OS name for OVF (Ubuntu | Ubuntu 64-bit)\n       osid: OS ID for OVF (93 | 94 )\n       diskname: name of disk file\n       disksize: size of virtual disk in bytes\n       mem: VM memory size in MB\n       cpus: # of virtual CPUs\n       vmname: Name for VM (default name when importing)\n       vmimfo: Brief description of VM for OVF'
    ovf = name + '.ovf'
    filesize = stat(diskname)[ST_SIZE]
    params = dict(osname=osname, osid=osid, diskname=diskname, filesize=filesize, disksize=disksize, name=name, mem=mem, cpus=cpus, vmname=vmname, vminfo=vminfo)
    xmltext = OVFTemplate % params
    with open(ovf, 'w+') as f:
        f.write(xmltext)
    return ovf

def qcow2size(qcow2):
    if False:
        i = 10
        return i + 15
    'Return virtual disk size (in bytes) of qcow2 image'
    output = check_output(['qemu-img', 'info', qcow2])
    try:
        assert 'format: qcow' in output
        bytes = int(re.findall('(\\d+) bytes', output)[0])
    except:
        raise Exception('Could not determine size of %s' % qcow2)
    return bytes

def build(flavor='raring32server', tests=None, pre='', post='', memory=1024):
    if False:
        while True:
            i = 10
    "Build a Mininet VM; return vmdk and vdisk size\n       tests: tests to run\n       pre: command line to run in VM before tests\n       post: command line to run in VM after tests\n       prompt: shell prompt (default '$ ')\n       memory: memory size in MB"
    global LogFile, Zip, Chown
    start = time()
    lstart = localtime()
    date = strftime('%y%m%d-%H-%M-%S', lstart)
    ovfdate = strftime('%y%m%d', lstart)
    dir = 'mn-%s-%s' % (flavor, date)
    if Branch:
        dir = 'mn-%s-%s-%s' % (Branch, flavor, date)
    try:
        os.mkdir(dir)
    except:
        raise Exception('Failed to create build directory %s' % dir)
    if Chown:
        run('chown %s %s' % (Chown, dir))
    os.chdir(dir)
    LogFile = open('build.log', 'w')
    log('* Logging to', abspath(LogFile.name))
    log('* Created working directory', dir)
    (image, kernel, initrd, partnum) = findBaseImage(flavor)
    basename = 'mininet-' + flavor
    volume = basename + '.qcow2'
    run('qemu-img create -f qcow2 -b %s %s' % (image, volume))
    log('* VM image for', flavor, 'created as', volume)
    if LogToConsole:
        logfile = stdout
    else:
        logfile = open(flavor + '.log', 'w+')
    log('* Logging results to', abspath(logfile.name))
    vm = boot(volume, kernel, initrd, logfile, memory=memory, partnum=partnum)
    version = interact(vm, tests=tests, pre=pre, post=post)
    size = qcow2size(volume)
    arch = archFor(flavor)
    vmdk = convert(volume, basename='mininet-vm-' + arch)
    if not SaveQCOW2:
        log('* Removing qcow2 volume', volume)
        os.remove(volume)
    log('* Converted VM image stored as', abspath(vmdk))
    ovfname = 'mininet-%s-%s-%s' % (version, ovfdate, OSVersion(flavor))
    (osname, osid) = OVFOSNameID(flavor)
    ovf = generateOVF(name=ovfname, osname=osname, osid=osid, diskname=vmdk, disksize=size)
    log('* Generated OVF descriptor file', ovf)
    if Zip:
        log('* Generating .zip file')
        run('zip %s-ovf.zip %s %s' % (ovfname, ovf, vmdk))
    end = time()
    elapsed = end - start
    log('* Results logged to', abspath(logfile.name))
    log('* Completed in %.2f seconds' % elapsed)
    log('* %s VM build DONE!!!!! :D' % flavor)
    os.chdir('..')

def runTests(vm, tests=None, pre='', post='', prompt=Prompt, uninstallNtpd=False):
    if False:
        print('Hello World!')
    'Run tests (list) in vm (pexpect object)'
    if uninstallNtpd:
        removeNtpd(vm)
        vm.expect(prompt)
    if Branch:
        checkOutBranch(vm, branch=Branch)
        vm.expect(prompt)
    if not tests:
        tests = []
    if pre:
        log('* Running command', pre)
        vm.sendline(pre)
        vm.expect(prompt)
    testfns = testDict()
    if tests:
        log('* Running tests')
    for test in tests:
        if test not in testfns:
            raise Exception('Unknown test: ' + test)
        log('* Running test', test)
        fn = testfns[test]
        fn(vm)
        vm.expect(prompt)
    if post:
        log('* Running post-test command', post)
        vm.sendline(post)
        vm.expect(prompt)

def getMininetVersion(vm):
    if False:
        for i in range(10):
            print('nop')
    'Run mn to find Mininet version in VM'
    vm.sendline('(cd ~/mininet; PYTHONPATH=. bin/mn --version)')
    vm.readline()
    version = vm.readline().strip()
    return version

def bootAndRun(image, prompt=Prompt, memory=1024, cpuCores=1, outputFile=None, runFunction=None, **runArgs):
    if False:
        return 10
    "Boot and test VM\n       tests: list of tests to run\n       pre: command line to run in VM before tests\n       post: command line to run in VM after tests\n       prompt: shell prompt (default '$ ')\n       memory: VM memory size in MB\n       cpuCores: number of CPU cores to use"
    bootTestStart = time()
    basename = path.basename(image)
    image = abspath(image)
    tmpdir = mkdtemp(prefix='test-' + basename)
    log('* Using tmpdir', tmpdir)
    cow = path.join(tmpdir, basename + '.qcow2')
    log('* Creating COW disk', cow)
    run('qemu-img create -f qcow2 -b %s %s' % (image, cow))
    log('* Extracting kernel and initrd')
    (kernel, initrd, partnum) = extractKernel(image, flavor=basename, imageDir=tmpdir)
    if LogToConsole:
        logfile = stdout
    else:
        logfile = NamedTemporaryFile(prefix=basename, suffix='.testlog', delete=False)
    log('* Logging VM output to', logfile.name)
    vm = boot(cow=cow, kernel=kernel, initrd=initrd, logfile=logfile, memory=memory, cpuCores=cpuCores, partnum=partnum)
    login(vm)
    log('* Waiting for prompt after login')
    vm.expect(prompt)
    if runFunction:
        runFunction(vm, **runArgs)
    log('* Shutting down')
    vm.sendline('sudo -n shutdown -h now ')
    log('* Waiting for shutdown')
    vm.wait()
    if outputFile:
        log('* Saving temporary image to %s' % outputFile)
        convert(cow, outputFile)
    log('* Removing temporary dir', tmpdir)
    srun('rm -rf ' + tmpdir)
    elapsed = time() - bootTestStart
    log('* Boot and test completed in %.2f seconds' % elapsed)

def buildFlavorString():
    if False:
        while True:
            i = 10
    'Return string listing valid build flavors'
    return 'valid build flavors: %s' % ' '.join(sorted(isoURLs))

def testDict():
    if False:
        return 10
    'Return dict of tests in this module'
    suffix = 'Test'
    trim = len(suffix)
    fdict = dict([(fname[:-trim], f) for (fname, f) in inspect.getmembers(modules[__name__], inspect.isfunction) if fname.endswith(suffix)])
    return fdict

def testString():
    if False:
        for i in range(10):
            print('nop')
    'Return string listing valid tests'
    tests = ['%s <%s>' % (name, func.__doc__) for (name, func) in testDict().items()]
    return 'valid tests: %s' % ', '.join(tests)

def parseArgs():
    if False:
        print('Hello World!')
    'Parse command line arguments and run'
    global LogToConsole, NoKVM, Branch, Zip, TIMEOUT, Forward, Chown
    parser = argparse.ArgumentParser(description='Mininet VM build script', epilog='')
    parser.add_argument('-v', '--verbose', action='store_true', help='send VM output to console rather than log file')
    parser.add_argument('-d', '--depend', action='store_true', help='install dependencies for this script')
    parser.add_argument('-l', '--list', action='store_true', help='list valid build flavors and tests')
    parser.add_argument('-c', '--clean', action='store_true', help='clean up leftover build junk (e.g. qemu-nbd)')
    parser.add_argument('-q', '--qcow2', action='store_true', help='save qcow2 image rather than deleting it')
    parser.add_argument('-n', '--nokvm', action='store_true', help="Don't use kvm - use tcg emulation instead")
    parser.add_argument('-m', '--memory', metavar='MB', type=int, default=1024, help='VM memory size in MB')
    parser.add_argument('-i', '--image', metavar='image', default=[], action='append', help='Boot and test an existing VM image')
    parser.add_argument('-t', '--test', metavar='test', default=[], action='append', help='specify a test to run; ' + testString())
    parser.add_argument('-w', '--timeout', metavar='timeout', type=int, default=0, help='set expect timeout')
    parser.add_argument('-r', '--run', metavar='cmd', default='', help='specify a command line to run before tests')
    parser.add_argument('-p', '--post', metavar='cmd', default='', help='specify a command line to run after tests')
    parser.add_argument('-b', '--branch', metavar='branch', help='branch to install and/or check out and test')
    parser.add_argument('flavor', nargs='*', help='VM flavor(s) to build; ' + buildFlavorString())
    parser.add_argument('-z', '--zip', action='store_true', help='archive .ovf and .vmdk into .zip file')
    parser.add_argument('-o', '--out', help='output file for test image (vmdk)')
    parser.add_argument('-f', '--forward', default=[], action='append', help='forward VM ports to local server, e.g. tcp:5555::22')
    parser.add_argument('-u', '--chown', metavar='user', help='specify an owner for build directory')
    args = parser.parse_args()
    if args.depend:
        depend()
    if args.list:
        print(buildFlavorString())
    if args.clean:
        cleanup()
    if args.verbose:
        LogToConsole = True
    if args.nokvm:
        NoKVM = True
    if args.branch:
        Branch = args.branch
    if args.zip:
        Zip = True
    if args.timeout:
        TIMEOUT = args.timeout
    if args.forward:
        Forward = args.forward
    if not args.test and (not args.run) and (not args.post):
        args.test = ['sanity', 'core']
    if args.chown:
        Chown = args.chown
    for flavor in args.flavor:
        if flavor not in isoURLs:
            print('Unknown build flavor:', flavor)
            print(buildFlavorString())
            break
        try:
            build(flavor, tests=args.test, pre=args.run, post=args.post, memory=args.memory)
        except Exception as e:
            log('* BUILD FAILED with exception: ', e)
            print_exc(e)
            exit(1)
    for image in args.image:
        bootAndRun(image, runFunction=runTests, tests=args.test, pre=args.run, post=args.post, memory=args.memory, outputFile=args.out, uninstallNtpd=True)
    if not (args.depend or args.list or args.clean or args.flavor or args.image):
        parser.print_help()
if __name__ == '__main__':
    parseArgs()