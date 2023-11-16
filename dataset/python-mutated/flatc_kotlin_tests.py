from flatc_test import *

class KotlinTests:

    def EnumValAttributes(self):
        if False:
            i = 10
            return i + 15
        flatc(['--kotlin', 'enum_val_attributes.fbs'])
        subject = assert_file_exists('ValAttributes.kt')
        assert_file_doesnt_contains(subject, 'val names : Array<String> = arrayOf("Val1", "Val2", "Val3")')
        assert_file_doesnt_contains(subject, 'fun name(e: Int) : String = names[e]')

    def EnumValAttributes_ReflectNames(self):
        if False:
            return 10
        flatc(['--kotlin', '--reflect-names', 'enum_val_attributes.fbs'])
        subject = assert_file_exists('ValAttributes.kt')
        assert_file_contains(subject, 'val names : Array<String> = arrayOf("Val1", "Val2", "Val3")')
        assert_file_contains(subject, 'fun name(e: Int) : String = names[e]')