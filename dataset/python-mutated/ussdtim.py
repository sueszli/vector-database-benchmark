import time
import sys
print('Welcome To fastrack USSD Banking Project...')
time.sleep(8)
bank_list = '\n1. Access Bank\n2. Fidelity Bank\n3. Guarantee Trust Bank\n4. Heritage Bank\n5. Polaris Bank\n6. Stanbic IBTC\n7. Unity Bank\n8. Wema Bank\n'
gen_bvn = ' '

def BVN_checker():
    if False:
        while True:
            i = 10
    global gen_bvn
    bvn = [str(i) for i in range(5)]
    gen_bvn = ''.join(bvn)

def open_acct():
    if False:
        while True:
            i = 10
    global gen_bvn
    print('Welcome to our online Account opening services.')
    print('loading...')
    temp_storage = []
    f_name = input('Enter your first name:')
    s_name = input('Enter your second name:')
    sex = input('Enter sex [M/F]:')
    BVN_checker()
    temp_storage.append(f_name)
    temp_storage.append(s_name)
    temp_storage.append(sex)
    temp_storage.append(gen_bvn)
    details = ' '.join(temp_storage)
    split_details = details.split(' ')
    print(split_details[0] + ' ' + split_details[1])
    print(split_details[2])
    print('Your bvn is :' + split_details[3])
    print('1. Press # to go back to options menu\n2. Press * to exit')
    bck = input(':')
    if bck == '#':
        options_menu()
    else:
        sys.exit()
    exit()

def upgrade_migrate():
    if False:
        return 10
    print('Welcome to our online Upgrade/Migration services.\n 1. Ugrade\n 2. Migrate')
    print('press # is go back to the Main Menu.')
    prompt = input('Enter preferred Choice:')
    if prompt == '1':
        time.sleep(5)
        print('Upgrading...')
        exit()
    elif prompt == '2':
        time.sleep(5)
        print('Migrating...')
        exit()
    elif prompt == '#':
        options_menu()
    else:
        sys.exit()

def balance():
    if False:
        return 10
    print('ACCOUNT\tBALANCE\n CHECKER')
    print('press # is go back to the Main Menu.')
    pin = input('Enter your 4 digit pin:')
    if len(pin) != 4:
        print('Make sure its a 4digit pin.')
        time.sleep(5)
        balance()
    elif pin.isdigit():
        time.sleep(5)
        print('Loading...')
        exit()
    elif pin == '#':
        options_menu()
    else:
        time.sleep(15)
        print('wrong pin')
        sys.exit()

def transf():
    if False:
        i = 10
        return i + 15
    print('1. Transfer self\n2. Transfer others')
    print('press # is go back to the Main Menu.')
    trnsf = input(':')
    if trnsf == '#':
        options_menu()
    elif trnsf == '1':
        time.sleep(5)
        print('Sending...')
        exit()
    elif trnsf == '2':
        time.sleep(5)
        num = int(input('Enter receivers mobile number:'))
        print('Transferring to', num)
        exit()
    elif trnsf.isdigit() != True:
        time.sleep(5)
        print('Not an option')
        sys.exit()
    elif trnsf.isdigit() and len(trnsf) > 2:
        time.sleep(5)
        print('wrong password.')
        sys.exit()
    else:
        time.sleep(10)
        print('An error has occurred')
        sys.exit()

def funds():
    if False:
        return 10
    time.sleep(3)
    print(bank_list)
    bnk = input('Select receipients Bank:')
    acc_num = input('Entet account number:')
    print('Sending to', acc_num)
    hash = input('1.Press # to go back to options menu\n2. Press * to go exit.')
    if hash == '#':
        options_menu()
    elif hash == '*':
        exit()
    else:
        sys.exit()

def options_menu():
    if False:
        print('Hello World!')
    print('1. Open Account\n2. Upgrade/Migrate\n3. Balance\n4. Transfer\n5. Funds')
    select_options = {'1': open_acct, '2': upgrade_migrate, '3': balance, '4': transf, '5': funds}
    choice = input('Enter an option:')
    if select_options.get(choice):
        select_options[choice]()
    else:
        sys.exit()

def exit():
    if False:
        for i in range(10):
            print('nop')
    exit = input('Do you wish to make another transaction [Y/N] :')
    if exit == 'N':
        sys.exit()
    elif exit == '#':
        options_menu()
    else:
        log_in()

def log_in():
    if False:
        for i in range(10):
            print('nop')
    try:
        a = 0
        while a < 3:
            a += 1
            USSD = input('ENTER USSD:')
            if USSD != '*919#':
                print('please re-enter USSD ...')
            else:
                print('Welcome to our online services how may we help you')
                options_menu()
                exit()
        else:
            time.sleep(10)
            print('checking discrepancies...')
            time.sleep(5)
            print('An error has occured.')
    except:
        sys.exit()
log_in()