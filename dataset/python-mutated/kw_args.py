def print_scores(**kw):
    if False:
        for i in range(10):
            print('nop')
    print('      Name  Score')
    print('------------------')
    for (name, score) in kw.items():
        print('%10s  %d' % (name, score))
    print()
print_scores(Adam=99, Lisa=88, Bart=77)
data = {'Adam Lee': 99, 'Lisa S': 88, 'F.Bart': 77}
print_scores(**data)

def print_info(name, *, gender, city='Beijing', age):
    if False:
        return 10
    print('Personal Info')
    print('---------------')
    print('   Name: %s' % name)
    print(' Gender: %s' % gender)
    print('   City: %s' % city)
    print('    Age: %s' % age)
    print()
print_info('Bob', gender='male', age=20)
print_info('Lisa', gender='female', city='Shanghai', age=18)