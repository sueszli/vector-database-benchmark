import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../python'))
import flatbuffers
import MyGame.Sample.Color
import MyGame.Sample.Equipment
import MyGame.Sample.Monster
import MyGame.Sample.Vec3
import MyGame.Sample.Weapon

def main():
    if False:
        return 10
    builder = flatbuffers.Builder(0)
    weapon_one = builder.CreateString('Sword')
    weapon_two = builder.CreateString('Axe')
    MyGame.Sample.Weapon.WeaponStart(builder)
    MyGame.Sample.Weapon.WeaponAddName(builder, weapon_one)
    MyGame.Sample.Weapon.WeaponAddDamage(builder, 3)
    sword = MyGame.Sample.Weapon.WeaponEnd(builder)
    MyGame.Sample.Weapon.WeaponStart(builder)
    MyGame.Sample.Weapon.WeaponAddName(builder, weapon_two)
    MyGame.Sample.Weapon.WeaponAddDamage(builder, 5)
    axe = MyGame.Sample.Weapon.WeaponEnd(builder)
    name = builder.CreateString('Orc')
    MyGame.Sample.Monster.MonsterStartInventoryVector(builder, 10)
    for i in reversed(range(0, 10)):
        builder.PrependByte(i)
    inv = builder.EndVector()
    MyGame.Sample.Monster.MonsterStartWeaponsVector(builder, 2)
    builder.PrependUOffsetTRelative(axe)
    builder.PrependUOffsetTRelative(sword)
    weapons = builder.EndVector()
    pos = MyGame.Sample.Vec3.CreateVec3(builder, 1.0, 2.0, 3.0)
    MyGame.Sample.Monster.MonsterStart(builder)
    MyGame.Sample.Monster.MonsterAddPos(builder, pos)
    MyGame.Sample.Monster.MonsterAddHp(builder, 300)
    MyGame.Sample.Monster.MonsterAddName(builder, name)
    MyGame.Sample.Monster.MonsterAddInventory(builder, inv)
    MyGame.Sample.Monster.MonsterAddColor(builder, MyGame.Sample.Color.Color().Red)
    MyGame.Sample.Monster.MonsterAddWeapons(builder, weapons)
    MyGame.Sample.Monster.MonsterAddEquippedType(builder, MyGame.Sample.Equipment.Equipment().Weapon)
    MyGame.Sample.Monster.MonsterAddEquipped(builder, axe)
    orc = MyGame.Sample.Monster.MonsterEnd(builder)
    builder.Finish(orc)
    buf = builder.Output()
    monster = MyGame.Sample.Monster.Monster.GetRootAsMonster(buf, 0)
    assert monster.Mana() == 150
    assert monster.Hp() == 300
    assert monster.Name() == b'Orc'
    assert monster.Color() == MyGame.Sample.Color.Color().Red
    assert monster.Pos().X() == 1.0
    assert monster.Pos().Y() == 2.0
    assert monster.Pos().Z() == 3.0
    for i in range(monster.InventoryLength()):
        assert monster.Inventory(i) == i
    expected_weapon_names = [b'Sword', b'Axe']
    expected_weapon_damages = [3, 5]
    for i in range(monster.WeaponsLength()):
        assert monster.Weapons(i).Name() == expected_weapon_names[i]
        assert monster.Weapons(i).Damage() == expected_weapon_damages[i]
    assert monster.EquippedType() == MyGame.Sample.Equipment.Equipment().Weapon
    if monster.EquippedType() == MyGame.Sample.Equipment.Equipment().Weapon:
        union_weapon = MyGame.Sample.Weapon.Weapon()
        union_weapon.Init(monster.Equipped().Bytes, monster.Equipped().Pos)
        assert union_weapon.Name() == b'Axe'
        assert union_weapon.Damage() == 5
    print('The FlatBuffer was successfully created and verified!')
if __name__ == '__main__':
    main()