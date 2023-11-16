# coding: utf-8
from sqlalchemy import BigInteger, Column, Date, DateTime, Float, Index, Integer, SmallInteger, String, Table, Text, text
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()
metadata = Base.metadata


class KqsAkcesoria(Base):
    __tablename__ = 'kqs_akcesoria'

    numer = Column(BigInteger, primary_key=True)
    produkt_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    grupa_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    kolejnosc = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsAkcesoriaGrupy(Base):
    __tablename__ = 'kqs_akcesoria_grupy'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(200), nullable=False, server_default=text("''"))
    nazwa_robocza = Column(String(200), nullable=False, server_default=text("''"))
    opis = Column(Text, nullable=False)
    kolejnosc = Column(Integer, nullable=False, server_default=text("'0'"))


t_kqs_akcesoria_kategorie = Table(
    'kqs_akcesoria_kategorie', metadata,
    Column('kategoria_id', BigInteger, nullable=False, index=True, server_default=text("'0'")),
    Column('grupa_id', BigInteger, nullable=False, index=True, server_default=text("'0'"))
)


t_kqs_akcesoria_produkty = Table(
    'kqs_akcesoria_produkty', metadata,
    Column('produkt_id', BigInteger, nullable=False, index=True, server_default=text("'0'")),
    Column('grupa_id', BigInteger, nullable=False, index=True, server_default=text("'0'"))
)


class KqsAktualnosci(Base):
    __tablename__ = 'kqs_aktualnosci'

    numer = Column(BigInteger, primary_key=True)
    temat = Column(String(255), nullable=False, server_default=text("''"))
    wstep = Column(Text, nullable=False)
    tresc = Column(Text, nullable=False)
    autor_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    aktywacja = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsBluemedia(Base):
    __tablename__ = 'kqs_bluemedia'

    numer = Column(BigInteger, primary_key=True)
    identyfikator = Column(String(40), nullable=False, server_default=text("''"))
    identyfikator_bm = Column(String(40), nullable=False, server_default=text("''"))
    kwota = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    waluta = Column(String(5), nullable=False, server_default=text("''"))
    status = Column(Integer, nullable=False, server_default=text("'0'"))
    raport = Column(String(255), nullable=False, server_default=text("''"))
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsCaraty(Base):
    __tablename__ = 'kqs_caraty'

    numer = Column(BigInteger, primary_key=True)
    status = Column(Integer, nullable=False, server_default=text("'0'"))
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsCechyGrupy(Base):
    __tablename__ = 'kqs_cechy_grupy'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(255), nullable=False, server_default=text("''"))


class KqsCechyGrupyOpcje(Base):
    __tablename__ = 'kqs_cechy_grupy_opcje'

    numer = Column(BigInteger, primary_key=True)
    opcja_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    grupa_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    kolejnosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))


class KqsDetektyw(Base):
    __tablename__ = 'kqs_detektyw'

    data = Column(BigInteger, primary_key=True, server_default=text("'0'"))
    URL = Column(String(255), nullable=False, server_default=text("''"))
    user_IP = Column(String(25), nullable=False, index=True, server_default=text("''"))
    user_HOST = Column(String(255), nullable=False, server_default=text("''"))
    user_REFERENT = Column(String(255), nullable=False, server_default=text("''"))
    user_AGENT = Column(String(255), nullable=False, server_default=text("''"))
    user_HUMAN = Column(Integer, nullable=False, index=True, server_default=text("'0'"))


class KqsDhl(Base):
    __tablename__ = 'kqs_dhl'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    dhl_id = Column(String(50), nullable=False, server_default=text("''"))
    dhl_data = Column(Date, nullable=False)
    dhl_kurier = Column(BigInteger, nullable=False, server_default=text("'0'"))
    dhl_status = Column(String(255), nullable=False, server_default=text("''"))
    dhl_pobranie = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    dhl_ubezpieczenie = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    dhl_szczegoly = Column(Text, nullable=False)


class KqsDhlOdbiory(Base):
    __tablename__ = 'kqs_dhl_odbiory'

    numer = Column(BigInteger, primary_key=True)
    dhl_order_id = Column(String(50), nullable=False, index=True, server_default=text("''"))
    dhl_data = Column(Date, nullable=False)
    dhl_od = Column(String(5), nullable=False, server_default=text("''"))
    dhl_do = Column(String(5), nullable=False, server_default=text("''"))
    dhl_info = Column(String(100), nullable=False, server_default=text("''"))


class KqsDostawcy(Base):
    __tablename__ = 'kqs_dostawcy'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(100), nullable=False, server_default=text("''"))
    kod_dostawcy = Column(String(25), nullable=False, server_default=text("''"))
    dane = Column(Text, nullable=False)


class KqsDostepnosc(Base):
    __tablename__ = 'kqs_dostepnosc'

    numer = Column(BigInteger, primary_key=True)
    wartosc = Column(String(255), nullable=False, server_default=text("''"))
    ceneo = Column(String(50), nullable=False, server_default=text("''"))
    nokaut = Column(String(50), nullable=False, server_default=text("''"))
    okazje_info = Column(String(50), nullable=False, server_default=text("''"))


class KqsDzialy(Base):
    __tablename__ = 'kqs_dzialy'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(25), nullable=False, server_default=text("''"))
    naglowek = Column(String(255), nullable=False, server_default=text("''"))
    zawartosc = Column(Text, nullable=False)
    miejsce = Column(Integer, nullable=False, server_default=text("'0'"))
    odnosnik = Column(Integer, nullable=False, server_default=text("'0'"))
    uprawnienia = Column(Integer, nullable=False, server_default=text("'0'"))
    title_strony = Column(String(200), nullable=False, server_default=text("''"))
    meta_tagi = Column(Text, nullable=False)


class KqsElementyJezykowe(Base):
    __tablename__ = 'kqs_elementy_jezykowe'

    numer = Column(BigInteger, primary_key=True)
    element_rodzaj = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    element_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    jezyk = Column(String(25), nullable=False, server_default=text("''"))
    wartosc = Column(Text, nullable=False, index=True)


class KqsElementyMenu(Base):
    __tablename__ = 'kqs_elementy_menu'

    numer = Column(BigInteger, primary_key=True)
    naglowek = Column(String(255), nullable=False, server_default=text("''"))
    zawartosc = Column(Text, nullable=False)
    strona_glowna = Column(Integer, nullable=False, server_default=text("'0'"))
    katalog = Column(Integer, nullable=False, server_default=text("'0'"))
    kolumna = Column(Integer, nullable=False, server_default=text("'0'"))
    kolor_naglowek = Column(String(15), nullable=False, server_default=text("''"))
    kolor_naglowek_tekst = Column(String(15), nullable=False, server_default=text("''"))
    kolor_naglowek_ramka = Column(String(15), nullable=False, server_default=text("''"))
    kolor_czesc_wlasciwa = Column(String(15), nullable=False, server_default=text("''"))
    kolor_czesc_wlasciwa_ramka = Column(String(15), nullable=False, server_default=text("''"))
    naglowek_tlo = Column(String(255), nullable=False, server_default=text("''"))
    naglowek_wysokosc = Column(Integer, nullable=False, server_default=text("'0'"))
    naglowek_mar_gora = Column(Integer, nullable=False, server_default=text("'0'"))
    tlo_odstep = Column(String(255), nullable=False, server_default=text("''"))
    odstep_wysokosc = Column(Integer, nullable=False, server_default=text("'0'"))
    naglowek_ramka_gora = Column(Integer, nullable=False, server_default=text("'0'"))
    naglowek_ramka_dol = Column(Integer, nullable=False, server_default=text("'0'"))
    naglowek_ramka_lewo = Column(Integer, nullable=False, server_default=text("'0'"))
    naglowek_ramka_prawo = Column(Integer, nullable=False, server_default=text("'0'"))
    czesc_ramka_dol = Column(Integer, nullable=False, server_default=text("'0'"))
    czesc_ramka_lewo = Column(Integer, nullable=False, server_default=text("'0'"))
    czesc_ramka_prawo = Column(Integer, nullable=False, server_default=text("'0'"))
    kolejnosc = Column(Integer, nullable=False, server_default=text("'0'"))
    element_szablon = Column(Text, nullable=False)


class KqsEnadawcaPrzesylki(Base):
    __tablename__ = 'kqs_enadawca_przesylki'

    numer = Column(BigInteger, primary_key=True)
    guid = Column(String(50), nullable=False, server_default=text("''"))
    nrlp = Column(String(50), nullable=False, server_default=text("''"))
    rodzaj = Column(String(50), nullable=False, server_default=text("''"))
    wartosc = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    pobranie = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    status = Column(Integer, nullable=False, server_default=text("'0'"))
    zbior_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsEnadawcaZbiory(Base):
    __tablename__ = 'kqs_enadawca_zbiory'

    numer = Column(BigInteger, primary_key=True)
    identyfikator = Column(String(50), nullable=False, server_default=text("''"))
    identyfikator_bufora = Column(String(50), nullable=False, server_default=text("''"))
    data_utworzenia = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data_wyslania = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsEraty(Base):
    __tablename__ = 'kqs_eraty'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    status = Column(Integer, nullable=False, server_default=text("'0'"))
    sekret = Column(String(20), nullable=False, server_default=text("''"))


class KqsFaktury(Base):
    __tablename__ = 'kqs_faktury'

    numer = Column(BigInteger, primary_key=True)
    nr_dokumentu = Column(BigInteger, nullable=False, server_default=text("'0'"))
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    data_wystawienia = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data_sprzedazy = Column(BigInteger, nullable=False, server_default=text("'0'"))
    forma_zaplaty = Column(String(50), nullable=False, server_default=text("''"))
    termin_zaplaty = Column(String(25), nullable=False, server_default=text("''"))
    miejsce_wystawienia = Column(String(50), nullable=False, server_default=text("''"))
    adnotacje = Column(Text, nullable=False)
    administrator_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    administrator_nazwa = Column(String(100), nullable=False, server_default=text("''"))
    data_duplikatu = Column(BigInteger, nullable=False, server_default=text("'0'"))
    fraza_podsumowujaca = Column(Integer, nullable=False, server_default=text("'0'"))
    marza = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsFerbuy(Base):
    __tablename__ = 'kqs_ferbuy'

    numer = Column(BigInteger, primary_key=True)
    ferbuy_amount = Column(BigInteger, nullable=False, server_default=text("'0'"))
    ferbuy_amount_received = Column(BigInteger, nullable=False, server_default=text("'0'"))
    ferbuy_currency = Column(String(5), nullable=False, server_default=text("''"))
    ferbuy_currency_received = Column(String(5), nullable=False, server_default=text("''"))
    ferbuy_extra = Column(String(50), nullable=False, unique=True, server_default=text("''"))
    ferbuy_transaction_id = Column(String(50), nullable=False, server_default=text("''"))
    ferbuy_status = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsFurgonetka(Base):
    __tablename__ = 'kqs_furgonetka'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    furgon_id = Column(String(30), nullable=False, server_default=text("''"))
    furgon_status = Column(String(255), nullable=False, server_default=text("''"))
    furgon_wysylka = Column(Integer, nullable=False, server_default=text("'0'"))
    furgon_lp = Column(String(50), nullable=False, server_default=text("''"))
    furgon_pobranie = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    furgon_ubezpieczenie = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    furgon_usluga = Column(String(50), nullable=False, server_default=text("''"))
    furgon_cena = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))


class KqsGaleria(Base):
    __tablename__ = 'kqs_galeria'

    numer = Column(BigInteger, primary_key=True)
    obraz = Column(String(100), nullable=False, server_default=text("''"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    wysokosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    szerokosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    rozszerzenie = Column(String(5), nullable=False, server_default=text("''"))


t_kqs_galeria_import = Table(
    'kqs_galeria_import', metadata,
    Column('plik', String(100), nullable=False, server_default=text("''")),
    Column('zaczep', BigInteger, nullable=False, server_default=text("'0'"))
)


class KqsGaleriaZaczepy(Base):
    __tablename__ = 'kqs_galeria_zaczepy'

    numer = Column(BigInteger, primary_key=True)
    obraz_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    kolejnosc = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsGlobkurier(Base):
    __tablename__ = 'kqs_globkurier'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    globkurier_numer = Column(String(30), nullable=False, server_default=text("''"))
    globkurier_status = Column(String(255), nullable=False, server_default=text("''"))
    globkurier_lp = Column(String(50), nullable=False, server_default=text("''"))
    globkurier_pobranie = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    globkurier_ubezpieczenie = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    globkurier_cena = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    globkurier_usluga = Column(String(50), nullable=False, server_default=text("''"))
    globkurier_punkt = Column(String(10), nullable=False, server_default=text("''"))
    globkurier_przewoznik = Column(String(25), nullable=False, server_default=text("''"))
    globkurier_dodatki = Column(String(255), nullable=False, server_default=text("''"))
    globkurier_data_nadania = Column(String(20), nullable=False, server_default=text("''"))
    globkurier_etykieta = Column(Integer, nullable=False, server_default=text("'0'"))
    globkurier_protokol = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsGl(Base):
    __tablename__ = 'kqs_gls'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    gls_id = Column(String(50), nullable=False, server_default=text("''"))
    gls_pn = Column(String(50), nullable=False, server_default=text("''"))
    gls_lp = Column(String(50), nullable=False, server_default=text("''"))
    gls_status = Column(String(255), nullable=False, server_default=text("''"))
    gls_pobranie = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))


t_kqs_grafika = Table(
    'kqs_grafika', metadata,
    Column('opcja', String(50), nullable=False, unique=True, server_default=text("''")),
    Column('wartosc', Text, nullable=False)
)


t_kqs_grafika_szablony = Table(
    'kqs_grafika_szablony', metadata,
    Column('szablon_pole', String(100), nullable=False, unique=True, server_default=text("''")),
    Column('szablon_wartosc', Text, nullable=False)
)


class KqsGrupyRabatowe(Base):
    __tablename__ = 'kqs_grupy_rabatowe'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(255), nullable=False, server_default=text("''"))


class KqsHistoriaKorespondencji(Base):
    __tablename__ = 'kqs_historia_korespondencji'

    numer = Column(BigInteger, primary_key=True)
    data = Column(String(15), nullable=False, server_default=text("''"))
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    admin_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    informacja = Column(Text, nullable=False)


class KqsHistoriaLogowania(Base):
    __tablename__ = 'kqs_historia_logowania'

    numer = Column(BigInteger, primary_key=True)
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    uzytkownik = Column(String(100), nullable=False, server_default=text("''"))
    rezultat = Column(Integer, nullable=False, server_default=text("'0'"))
    IP_adres = Column(String(20), nullable=False, server_default=text("''"))


class KqsImportSzablony(Base):
    __tablename__ = 'kqs_import_szablony'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(255), nullable=False, server_default=text("''"))
    cykl = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    identyfikator = Column(String(50), nullable=False, server_default=text("''"))


class KqsImportSzablonyPola(Base):
    __tablename__ = 'kqs_import_szablony_pola'

    numer = Column(BigInteger, primary_key=True)
    szablon_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    pole = Column(String(50), nullable=False, server_default=text("''"))
    kolejnosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))


class KqsIndywidualni(Base):
    __tablename__ = 'kqs_indywidualni'

    numer = Column(BigInteger, primary_key=True)
    imie = Column(String(50), nullable=False, server_default=text("''"))
    nazwisko = Column(String(50), nullable=False, server_default=text("''"))
    firma = Column(String(200), nullable=False, server_default=text("''"))
    nip = Column(String(25), nullable=False, server_default=text("''"))
    ulica = Column(String(100), nullable=False, server_default=text("''"))
    dom = Column(String(20), nullable=False, server_default=text("''"))
    kod = Column(String(10), nullable=False, server_default=text("''"))
    miasto = Column(String(100), nullable=False, server_default=text("''"))
    wojewodztwo = Column(String(100), nullable=False, server_default=text("''"))
    telefon = Column(String(25), nullable=False, server_default=text("''"))
    email = Column(String(100), nullable=False, server_default=text("''"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    wysylka_odbiorca = Column(String(200), nullable=False, server_default=text("''"))
    wysylka_ulica = Column(String(100), nullable=False, server_default=text("''"))
    wysylka_dom = Column(String(20), nullable=False, server_default=text("''"))
    wysylka_kod = Column(String(10), nullable=False, server_default=text("''"))
    wysylka_miasto = Column(String(100), nullable=False, server_default=text("''"))
    regon = Column(String(20), nullable=False, server_default=text("''"))
    pesel = Column(String(20), nullable=False, server_default=text("''"))
    strona_www = Column(String(50), nullable=False, server_default=text("''"))
    gadu_gadu = Column(String(10), nullable=False, server_default=text("''"))
    skype = Column(String(35), nullable=False, server_default=text("''"))
    zgoda_1 = Column(Integer, nullable=False, server_default=text("'0'"))
    zgoda_2 = Column(Integer, nullable=False, server_default=text("'0'"))
    zgoda_3 = Column(Integer, nullable=False, server_default=text("'0'"))
    hurtownik = Column(Integer, nullable=False, server_default=text("'0'"))
    grupa_rabatowa = Column(BigInteger, nullable=False, server_default=text("'0'"))
    termin_przelewu = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsInpost(Base):
    __tablename__ = 'kqs_inpost'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    inpost_kod_przesylki = Column(String(50), nullable=False, server_default=text("''"))
    inpost_odbiorca = Column(String(50), nullable=False, server_default=text("''"))
    inpost_telefon = Column(String(50), nullable=False, server_default=text("''"))
    inpost_paczkomat = Column(String(10), nullable=False, server_default=text("''"))
    inpost_rozmiar = Column(String(5), nullable=False, server_default=text("''"))
    inpost_ubezpieczenie = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    inpost_pobranie = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    inpost_etykieta = Column(Integer, nullable=False, server_default=text("'0'"))
    inpost_potwierdzenie = Column(Integer, nullable=False, server_default=text("'0'"))
    inpost_wysylka = Column(Integer, nullable=False, server_default=text("'0'"))
    inpost_status = Column(String(30), nullable=False, server_default=text("''"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsKategorie(Base):
    __tablename__ = 'kqs_kategorie'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(100), nullable=False, server_default=text("''"))
    aktywne = Column(Integer, nullable=False, server_default=text("'0'"))
    kat_matka = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    kolejnosc = Column(BigInteger, nullable=False, server_default=text("'0'"))
    wizytowka = Column(Text, nullable=False)
    meta_tagi = Column(Text, nullable=False)
    archiwalna = Column(Integer, nullable=False, server_default=text("'0'"))
    opis_nadrzedny = Column(Text, nullable=False)
    title_strony = Column(String(200), nullable=False, server_default=text("''"))
    rozwiniecie = Column(Integer, nullable=False, server_default=text("'0'"))
    urzadzenie = Column(Integer, nullable=False, server_default=text("'0'"))
    obraz = Column(String(25), nullable=False, server_default=text("''"))
    ikona = Column(String(25), nullable=False, server_default=text("''"))
    raty = Column(Integer, nullable=False, server_default=text("'0'"))
    caraty_id = Column(String(10), nullable=False, server_default=text("''"))


class KqsKategorieBkp(Base):
    __tablename__ = 'kqs_kategorie_bkp'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(100), nullable=False, server_default=text("''"))
    aktywne = Column(Integer, nullable=False, server_default=text("'0'"))
    kat_matka = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    kolejnosc = Column(BigInteger, nullable=False, server_default=text("'0'"))
    wizytowka = Column(Text, nullable=False)
    meta_tagi = Column(Text, nullable=False)
    archiwalna = Column(Integer, nullable=False, server_default=text("'0'"))
    opis_nadrzedny = Column(Text, nullable=False)


t_kqs_kategorie_opcje = Table(
    'kqs_kategorie_opcje', metadata,
    Column('opcja_id', BigInteger, nullable=False, index=True, server_default=text("'0'")),
    Column('kategoria_id', BigInteger, nullable=False, index=True, server_default=text("'0'"))
)


t_kqs_kategorie_opcje_atrybuty = Table(
    'kqs_kategorie_opcje_atrybuty', metadata,
    Column('opcja_id', BigInteger, nullable=False, index=True, server_default=text("'0'")),
    Column('kategoria_id', BigInteger, nullable=False, index=True, server_default=text("'0'"))
)


class KqsKodyKwotowe(Base):
    __tablename__ = 'kqs_kody_kwotowe'

    numer = Column(BigInteger, primary_key=True)
    kod_rabatowy = Column(String(35), nullable=False, unique=True, server_default=text("''"))
    rabat = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    wykorzystano = Column(Integer, nullable=False, server_default=text("'0'"))
    data_dodania = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data_wygasniecia = Column(BigInteger, nullable=False, server_default=text("'0'"))
    kwota_minimalna = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))


class KqsKodyRabatowe(Base):
    __tablename__ = 'kqs_kody_rabatowe'

    numer = Column(BigInteger, primary_key=True)
    kod_rabatowy = Column(String(35), nullable=False, server_default=text("''"))
    rabat = Column(Float(4, True), nullable=False, server_default=text("'0.00'"))
    produkt_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    kategoria_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data_dodania = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data_wygasniecia = Column(BigInteger, nullable=False, server_default=text("'0'"))
    producent_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    granica = Column(Integer, nullable=False, server_default=text("'0'"))
    wykorzystano = Column(Integer, nullable=False, server_default=text("'0'"))
    kwota_minimalna = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))


class KqsKomunikaty(Base):
    __tablename__ = 'kqs_komunikaty'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(255), nullable=False, server_default=text("''"))
    temat = Column(String(100), nullable=False, server_default=text("''"))
    komunikat = Column(Text, nullable=False)
    format = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsKomunikatyDodatkowe(Base):
    __tablename__ = 'kqs_komunikaty_dodatkowe'

    numer = Column(BigInteger, primary_key=True)
    komunikat_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    nazwa = Column(String(255), nullable=False, server_default=text("''"))
    temat = Column(String(100), nullable=False, server_default=text("''"))
    komunikat = Column(Text, nullable=False)
    format = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsKomunikatyZalaczniki(Base):
    __tablename__ = 'kqs_komunikaty_zalaczniki'

    numer = Column(BigInteger, primary_key=True)
    komunikat_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    nazwa = Column(String(100), nullable=False, server_default=text("''"))
    plik = Column(String(100), nullable=False, server_default=text("''"))


t_kqs_konfiguracja = Table(
    'kqs_konfiguracja', metadata,
    Column('opcja', String(50), nullable=False, unique=True, server_default=text("''")),
    Column('wartosc', Text, nullable=False)
)


t_kqs_konwersja = Table(
    'kqs_konwersja', metadata,
    Column('data', BigInteger, nullable=False, index=True, server_default=text("'0'")),
    Column('zamowienie_id', BigInteger, nullable=False, index=True, server_default=text("'0'")),
    Column('URL', String(255), nullable=False, server_default=text("''"))
)


class KqsKorekty(Base):
    __tablename__ = 'kqs_korekty'

    numer = Column(BigInteger, primary_key=True)
    nr_dokumentu = Column(BigInteger, nullable=False, server_default=text("'0'"))
    fv_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    korekta = Column(Text, nullable=False)
    data_wystawienia = Column(BigInteger, nullable=False, server_default=text("'0'"))
    forma_zaplaty = Column(String(50), nullable=False, server_default=text("''"))
    termin_zaplaty = Column(String(25), nullable=False, server_default=text("''"))
    miejsce_wystawienia = Column(String(50), nullable=False, server_default=text("''"))
    adnotacje = Column(Text, nullable=False)
    powod_korekty = Column(Text, nullable=False)
    administrator_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    administrator_nazwa = Column(String(100), nullable=False, server_default=text("''"))
    data_duplikatu = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsKorektyProdukty(Base):
    __tablename__ = 'kqs_korekty_produkty'

    numer = Column(BigInteger, primary_key=True)
    korekta_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt = Column(String(200), nullable=False, server_default=text("''"))
    ilosc = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    jm = Column(String(50), nullable=False, server_default=text("''"))
    cena = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    podatek = Column(Integer, nullable=False, server_default=text("'0'"))
    pkwiu = Column(String(50), nullable=False, server_default=text("''"))
    rabat = Column(Float(4, True), nullable=False, server_default=text("'0.00'"))


class KqsKorespondencjaOdpowiedzi(Base):
    __tablename__ = 'kqs_korespondencja_odpowiedzi'

    numer = Column(BigInteger, primary_key=True)
    wiadomosc_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odpowiedz = Column(Text, nullable=False)
    data_wyslania = Column(BigInteger, nullable=False, server_default=text("'0'"))
    administrator_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    administrator_nazwa = Column(String(255), nullable=False, server_default=text("''"))


class KqsKorespondencjaPytania(Base):
    __tablename__ = 'kqs_korespondencja_pytania'

    numer = Column(BigInteger, primary_key=True)
    jezyk = Column(String(100), nullable=False, server_default=text("''"))
    formularz = Column(Integer, nullable=False, server_default=text("'0'"))
    produkt_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    produkt_nazwa = Column(String(255), nullable=False, server_default=text("''"))
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    klient_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    klient_nazwa = Column(String(255), nullable=False, server_default=text("''"))
    klient_email = Column(String(100), nullable=False, server_default=text("''"))
    klient_telefon = Column(String(100), nullable=False, server_default=text("''"))
    klient_host = Column(String(100), nullable=False, server_default=text("''"))
    klient_ip = Column(String(100), nullable=False, server_default=text("''"))
    temat = Column(String(255), nullable=False, server_default=text("''"))
    wiadomosc = Column(Text, nullable=False)
    data_nadejscia = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsKorespondencjaWiadomosci(Base):
    __tablename__ = 'kqs_korespondencja_wiadomosci'

    numer = Column(BigInteger, primary_key=True)
    klient_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    klient_nazwa = Column(String(255), nullable=False, server_default=text("''"))
    klient_email = Column(String(100), nullable=False, server_default=text("''"))
    klient_telefon = Column(String(100), nullable=False, server_default=text("''"))
    wiadomosc = Column(Text, nullable=False)
    data_wyslania = Column(BigInteger, nullable=False, server_default=text("'0'"))
    administrator_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    administrator_nazwa = Column(String(255), nullable=False, server_default=text("''"))


class KqsKoszyk(Base):
    __tablename__ = 'kqs_koszyk'

    id = Column(BigInteger, primary_key=True)
    p_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    ilosc = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    u_id = Column(String(50), nullable=False, server_default=text("''"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    u_reg = Column(BigInteger, nullable=False, server_default=text("'0'"))
    atrybuty = Column(String(255), nullable=False, server_default=text("''"))
    kod_rabatowy = Column(String(40), nullable=False, server_default=text("''"))
    atrybuty_bezwzgledne = Column(String(255), nullable=False, server_default=text("''"))


class KqsKoszykAdmin(Base):
    __tablename__ = 'kqs_koszyk_admin'

    id = Column(BigInteger, primary_key=True)
    p_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    ilosc = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    cena = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    podatek = Column(Integer, nullable=False, server_default=text("'0'"))
    rabat = Column(Float(4, True), nullable=False, server_default=text("'0.00'"))
    u_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    dzial = Column(Integer, nullable=False, server_default=text("'0'"))
    element = Column(BigInteger, nullable=False, server_default=text("'0'"))
    atrybuty = Column(String(255), nullable=False, server_default=text("''"))
    atrybuty_bezwzgledne = Column(String(255), nullable=False, server_default=text("''"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    kod_rabatowy = Column(String(50), nullable=False, server_default=text("''"))


class KqsKraje(Base):
    __tablename__ = 'kqs_kraje'

    numer = Column(BigInteger, primary_key=True)
    kraj = Column(String(100), nullable=False, server_default=text("''"))
    domyslny = Column(Integer, nullable=False, server_default=text("'0'"))
    kolejnosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    iso_alfa2 = Column(String(4), nullable=False, server_default=text("''"))
    iso_alfa3 = Column(String(4), nullable=False, server_default=text("''"))
    iso_num = Column(String(4), nullable=False, server_default=text("''"))
    bezplatna_dostawa = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsKurjerzy(Base):
    __tablename__ = 'kqs_kurjerzy'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    kurjerzy_status = Column(Integer, nullable=False, server_default=text("'0'"))
    kurjerzy_kod_przesylki = Column(String(50), nullable=False, server_default=text("''"))
    kurjerzy_waga = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    kurjerzy_wysokosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    kurjerzy_szerokosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    kurjerzy_dlugosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    kurjerzy_zawartosc = Column(String(50), nullable=False, server_default=text("''"))
    kurjerzy_wartosc = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    kurjerzy_ubezpieczenie = Column(Integer, nullable=False, server_default=text("'0'"))
    kurjerzy_pobranie = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


t_kqs_magazyn_historia = Table(
    'kqs_magazyn_historia', metadata,
    Column('produkt_id', BigInteger, nullable=False, index=True, server_default=text("'0'")),
    Column('data', BigInteger, nullable=False, server_default=text("'0'")),
    Column('stan', Float(asdecimal=True), nullable=False, server_default=text("'0'"))
)


class KqsMagazynPz(Base):
    __tablename__ = 'kqs_magazyn_pz'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    dostawca_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    dostawca_nazwa = Column(String(200), nullable=False, server_default=text("''"))
    nr_fv = Column(String(30), nullable=False, server_default=text("''"))
    nr_dokumentu = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data_wystawienia = Column(BigInteger, nullable=False, server_default=text("'0'"))
    adnotacje = Column(Text, nullable=False)
    admin_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    admin_nazwa = Column(String(100), nullable=False, server_default=text("''"))
    rabat_model = Column(Integer, nullable=False, server_default=text("'0'"))
    rodzaj_cen = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsMagazynPzProdukty(Base):
    __tablename__ = 'kqs_magazyn_pz_produkty'

    numer = Column(BigInteger, primary_key=True)
    dokument_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt_nazwa = Column(String(200), nullable=False, server_default=text("''"))
    ilosc = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    jm = Column(String(50), nullable=False, server_default=text("''"))
    cena = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    podatek = Column(Integer, nullable=False, server_default=text("'0'"))
    atrybuty = Column(Text, nullable=False)
    atrybuty_magazyn = Column(Text, nullable=False)
    pkwiu = Column(String(50), nullable=False, server_default=text("''"))
    kod_produktu = Column(String(50), nullable=False, server_default=text("''"))
    kod_dostawcy = Column(String(50), nullable=False, server_default=text("''"))
    kod_producenta = Column(String(50), nullable=False, server_default=text("''"))
    kod_kreskowy = Column(String(25), nullable=False, server_default=text("''"))
    kod_plu = Column(String(25), nullable=False, server_default=text("''"))
    rabat = Column(Float(4, True), nullable=False, server_default=text("'0.00'"))


class KqsMagazynWz(Base):
    __tablename__ = 'kqs_magazyn_wz'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    nr_dokumentu = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data_wystawienia = Column(BigInteger, nullable=False, server_default=text("'0'"))
    adnotacje = Column(Text, nullable=False)
    admin_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    admin_nazwa = Column(String(100), nullable=False, server_default=text("''"))
    rabat_model = Column(Integer, nullable=False, server_default=text("'0'"))
    rodzaj_cen = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsMagazynWzProdukty(Base):
    __tablename__ = 'kqs_magazyn_wz_produkty'

    numer = Column(BigInteger, primary_key=True)
    dokument_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt_nazwa = Column(String(200), nullable=False, server_default=text("''"))
    ilosc = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    jm = Column(String(50), nullable=False, server_default=text("''"))
    cena = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    podatek = Column(Integer, nullable=False, server_default=text("'0'"))
    atrybuty = Column(Text, nullable=False)
    atrybuty_magazyn = Column(Text, nullable=False)
    pkwiu = Column(String(50), nullable=False, server_default=text("''"))
    kod_produktu = Column(String(50), nullable=False, server_default=text("''"))
    kod_dostawcy = Column(String(50), nullable=False, server_default=text("''"))
    kod_producenta = Column(String(50), nullable=False, server_default=text("''"))
    kod_kreskowy = Column(String(25), nullable=False, server_default=text("''"))
    kod_plu = Column(String(25), nullable=False, server_default=text("''"))
    rabat = Column(Float(4, True), nullable=False, server_default=text("'0.00'"))


class KqsMbankRaty(Base):
    __tablename__ = 'kqs_mbank_raty'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, unique=True, server_default=text("'0'"))
    nr_wniosku = Column(String(32), nullable=False, server_default=text("''"))


class KqsMyCategoryAllegro(Base):
    __tablename__ = 'kqs_my_category_allegro'
    __table_args__ = (
        Index('country_id_2', 'country_id', 'parent_id'),
    )

    id = Column(Integer, primary_key=True, nullable=False)
    country_id = Column(Integer, primary_key=True, nullable=False, index=True)
    name = Column(String(50, 'utf8_polish_ci'), nullable=False)
    parent_id = Column(Integer, nullable=False)
    position = Column(Integer, nullable=False)


class KqsMyConfiguration(Base):
    __tablename__ = 'kqs_my_configuration'
    __table_args__ = (
        Index('group_cname', 'group', 'config_name', 'param_name', unique=True),
    )

    id = Column(BigInteger, primary_key=True, nullable=False)
    group = Column(String(32), primary_key=True, nullable=False, server_default=text("''"))
    config_name = Column(String(64), primary_key=True, nullable=False, server_default=text("''"))
    param_name = Column(String(64), primary_key=True, nullable=False, server_default=text("''"))
    param_value = Column(String, nullable=False)


class KqsMyProductMap(Base):
    __tablename__ = 'kqs_my_product_map'
    __table_args__ = (
        Index('product_id', 'global_id', 'platform'),
        Index('country_id', 'platform', 'foreign_id')
    )

    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, nullable=False, server_default=text("'0'"))
    platform = Column(String(64, 'utf8_polish_ci'), nullable=False)
    foreign_id = Column(String(64, 'utf8_polish_ci'), nullable=False)
    foreign_quantity = Column(Float(asdecimal=True), nullable=False)
    foreign_sold = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    foreign_price = Column(Float(asdecimal=True), nullable=False)
    title = Column(String(50, 'utf8_polish_ci'), nullable=False)
    global_id = Column(String(64, 'utf8_polish_ci'), nullable=False)
    global_ext = Column(String(64, 'utf8_polish_ci'), nullable=False)
    scheme_id = Column(String(64, 'utf8_polish_ci'), nullable=False)
    status = Column(Integer, nullable=False, server_default=text("'0'"))
    sell_again = Column(Integer, nullable=False, server_default=text("'0'"))
    date = Column(BigInteger, nullable=False)


class KqsMyScheme(Base):
    __tablename__ = 'kqs_my_scheme'
    __table_args__ = (
        Index('platform_2', 'platform', 'language'),
        Index('platform', 'global_id', 'platform', 'language')
    )

    id = Column(BigInteger, primary_key=True)
    global_id = Column(String(64, 'utf8_polish_ci'), nullable=False)
    global_ext = Column(String(64, 'utf8_polish_ci'), nullable=False)
    language = Column(String(16, 'utf8_polish_ci'), nullable=False)
    currency = Column(String(16, 'utf8_polish_ci'), nullable=False)
    profile_id = Column(Integer, nullable=False, server_default=text("'0'"))
    platform = Column(String(64, 'utf8_polish_ci'), nullable=False)
    offer_type = Column(Integer)
    title = Column(String(64, 'utf8_polish_ci'), nullable=False)
    foreign_cat_id = Column(String(64, 'utf8_polish_ci'), nullable=False)
    description = Column(String(512, 'utf8_polish_ci'), nullable=False)
    quantity = Column(Integer, nullable=False)
    cost = Column(Float(asdecimal=True), nullable=False)
    min_price = Column(Float(asdecimal=True), nullable=False)
    start_price = Column(Float(asdecimal=True), nullable=False)
    buynow_price = Column(Float(asdecimal=True), nullable=False)
    photo = Column(String(256, 'utf8_polish_ci'), nullable=False)
    template = Column(String(64, 'utf8_polish_ci'), nullable=False)
    path = Column(String(256), nullable=False, unique=True)
    running = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    queued = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    scheduled = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    date = Column(BigInteger, nullable=False)
    status = Column(Integer, nullable=False)


class KqsMyShipping(Base):
    __tablename__ = 'kqs_my_shipping'
    __table_args__ = (
        Index('carrier_shipping_no', 'carrier', 'shipping_no', unique=True),
    )

    id = Column(BigInteger, primary_key=True)
    profile_id = Column(Integer, nullable=False, server_default=text("'0'"))
    source = Column(String(128, 'utf8_polish_ci'), nullable=False, server_default=text("'local'"))
    order_id = Column(String(64, 'utf8_polish_ci'), nullable=False)
    carrier = Column(String(64, 'utf8_polish_ci'), nullable=False)
    carrier_id = Column(String(64, 'utf8_polish_ci'), nullable=False)
    shipping_no = Column(String(64, 'utf8_polish_ci'), nullable=False)
    protocol_no = Column(String(64, 'utf8_polish_ci'), nullable=False)
    receiver = Column(String(512, 'utf8_polish_ci'), nullable=False)
    extra = Column(String(64, 'utf8_polish_ci'), nullable=False)
    created = Column(BigInteger, nullable=False)
    dispatched = Column(BigInteger, nullable=False)
    delivered = Column(BigInteger, nullable=False)
    status = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsMySynchro(Base):
    __tablename__ = 'kqs_my_synchro'
    __table_args__ = (
        Index('scheme', 'scheme', 'product_id', 'status', 'date'),
        Index('session', 'session', 'scheme', 'status')
    )

    id = Column(Integer, primary_key=True)
    session = Column(String(32, 'utf8_polish_ci'), nullable=False)
    type = Column(Integer, nullable=False, server_default=text("'1'"))
    scheme = Column(String(32, 'utf8_polish_ci'), nullable=False)
    product_id = Column(String(32, 'utf8_polish_ci'), nullable=False)
    date = Column(Integer, nullable=False)
    status = Column(Integer, nullable=False)
    info = Column(String(255, 'utf8_polish_ci'), nullable=False)


class KqsMySynchroProduct(Base):
    __tablename__ = 'kqs_my_synchro_product'
    __table_args__ = (
        Index('producer_id', 'profile_id', 'platform', 'producer_id'),
    )

    profile_id = Column(Integer, primary_key=True, nullable=False, server_default=text("'0'"))
    platform = Column(String(64, 'utf8_polish_ci'), primary_key=True, nullable=False)
    id = Column(String(64, 'utf8_polish_ci'), primary_key=True, nullable=False)
    variant_id = Column(String(64, 'utf8_polish_ci'), primary_key=True, nullable=False, server_default=text("''"))
    code = Column(String(32, 'utf8_polish_ci'), nullable=False)
    variant_code = Column(String(32, 'utf8_polish_ci'))
    name = Column(String(50, 'utf8_polish_ci'), nullable=False)
    producer_id = Column(String(32), nullable=False, server_default=text("'0'"))
    quantity = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    base_quantity = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    price = Column(Float(asdecimal=True), nullable=False)
    discount_price = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    discount = Column(Integer, nullable=False, server_default=text("'0'"))
    currency_code = Column(String(5))
    main_category = Column(String(32, 'utf8_polish_ci'), nullable=False)
    image = Column(String(128))
    active = Column(Integer, nullable=False, server_default=text("'0'"))
    updated = Column(Integer, nullable=False)
    date_add = Column(DateTime)
    date_update = Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))
    date_delete = Column(DateTime)
    assigned = Column(Integer, nullable=False, server_default=text("'1'"))


class KqsMyTask(Base):
    __tablename__ = 'kqs_my_task'

    id = Column(BigInteger, primary_key=True)
    group = Column(String(64, 'utf8_polish_ci'), nullable=False)
    profile_id = Column(Integer, nullable=False)
    doer = Column(String(64, 'utf8_polish_ci'), nullable=False)
    prms = Column(String(128, 'utf8_polish_ci'), nullable=False)
    tstart = Column(Integer, nullable=False)
    tstop = Column(Integer, nullable=False)
    interval = Column(Integer, nullable=False)
    order = Column(Integer, nullable=False)
    status = Column(SmallInteger, nullable=False)
    date = Column(Integer, nullable=False)
    info = Column(String(256, 'utf8_polish_ci'), nullable=False, server_default=text("''"))


class KqsMyTransactionAdeal(Base):
    __tablename__ = 'kqs_my_transaction_adeal'
    __table_args__ = (
        Index('foreign_id', 'foreign_id', 'buyer_id'),
        Index('source_id', 'source', 'foreign_id')
    )

    deal_id = Column(BigInteger, primary_key=True)
    source = Column(String(64, 'utf8_polish_ci'), nullable=False)
    foreign_id = Column(String(64, 'utf8_polish_ci'), nullable=False)
    foreign_name = Column(String(128, 'utf8_polish_ci'), nullable=False)
    buyer_id = Column(String(64, 'utf8_polish_ci'), nullable=False)
    buyer_login = Column(String(32, 'utf8_polish_ci'), nullable=False)
    global_id = Column(String(64, 'utf8_polish_ci'), nullable=False)
    user_login = Column(String(32, 'utf8_polish_ci'), nullable=False)
    date = Column(BigInteger, nullable=False)
    status = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsMyTransactionMap(Base):
    __tablename__ = 'kqs_my_transaction_map'
    __table_args__ = (
        Index('source_id', 'source', 'foreign_id', unique=True),
    )

    id = Column(BigInteger, primary_key=True)
    source = Column(String(64, 'utf8_polish_ci'), nullable=False)
    foreign_id = Column(String(64, 'utf8_polish_ci'), nullable=False)
    global_id = Column(String(64, 'utf8_polish_ci'), nullable=False)
    date = Column(BigInteger, nullable=False)
    status = Column(Integer, nullable=False, server_default=text("'0'"))
    timestamp = Column(DateTime, nullable=False, server_default=text("'0000-00-00 00:00:00'"))


class KqsNewsletter(Base):
    __tablename__ = 'kqs_newsletter'

    numer = Column(BigInteger, primary_key=True)
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    temat = Column(String(200), nullable=False, server_default=text("''"))
    nowosci = Column(Integer, nullable=False, server_default=text("'0'"))
    tresc = Column(Text, nullable=False)
    grupa = Column(BigInteger, nullable=False, server_default=text("'0'"))
    format = Column(Integer, nullable=False, server_default=text("'0'"))
    stan = Column(BigInteger, nullable=False, server_default=text("'0'"))
    wyslany = Column(Integer, nullable=False, server_default=text("'0'"))
    opiekun = Column(String(50), nullable=False, server_default=text("''"))
    opiekun_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    branza = Column(String(50), nullable=False, server_default=text("''"))
    branza_id = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsNotatkiCrm(Base):
    __tablename__ = 'kqs_notatki_crm'

    numer = Column(BigInteger, primary_key=True)
    klient_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    klient_typ = Column(Integer, nullable=False, server_default=text("'1'"))
    notatka = Column(Text, nullable=False)
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    admin_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    ost_admin_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    ost_admin_data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsOgladane(Base):
    __tablename__ = 'kqs_ogladane'

    numer = Column(BigInteger, primary_key=True)
    p_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    u_id = Column(String(50), nullable=False, index=True, server_default=text("''"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsParagony(Base):
    __tablename__ = 'kqs_paragony'

    numer = Column(BigInteger, primary_key=True)
    nr_dokumentu = Column(BigInteger, nullable=False, server_default=text("'0'"))
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    data_wystawienia = Column(BigInteger, nullable=False, server_default=text("'0'"))
    adnotacje = Column(Text, nullable=False)
    administrator_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    administrator_nazwa = Column(String(100), nullable=False, server_default=text("''"))


class KqsPaybynet(Base):
    __tablename__ = 'kqs_paybynet'

    numer = Column(BigInteger, primary_key=True)
    amount = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    date_valid = Column(String(19), nullable=False, server_default=text("''"))
    id_trans = Column(String(10), nullable=False, unique=True, server_default=text("''"))
    status = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsPaypal(Base):
    __tablename__ = 'kqs_paypal'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    status = Column(Integer, nullable=False, server_default=text("'0'"))
    sekret = Column(String(20), nullable=False, server_default=text("''"))
    kwota = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    waluta = Column(String(5), nullable=False, server_default=text("''"))
    paypal_kwota = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    paypal_waluta = Column(String(5), nullable=False, server_default=text("''"))
    paypal_platnik_id = Column(String(20), nullable=False, server_default=text("''"))
    paypal_platnik_email = Column(String(100), nullable=False, server_default=text("''"))
    paypal_correlation_id = Column(String(50), nullable=False, server_default=text("''"))
    paypal_transaction_id = Column(String(50), nullable=False, server_default=text("''"))
    paypal_status = Column(String(50), nullable=False, server_default=text("''"))
    paypal_raport = Column(String(255), nullable=False, server_default=text("''"))


class KqsPersonalia(Base):
    __tablename__ = 'kqs_personalia'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(100), nullable=False, unique=True, server_default=text("''"))


class KqsPkwid(Base):
    __tablename__ = 'kqs_pkwid'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    pkwid_id = Column(String(30), nullable=False, server_default=text("''"))
    pkwid_status = Column(String(255), nullable=False, server_default=text("''"))
    pkwid_labelNumber = Column(String(50), nullable=False, server_default=text("''"))
    pkwid_courierOrderNumber = Column(String(50), nullable=False, server_default=text("''"))
    pkwid_codAmount = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    pkwid_grossPrice = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    pkwid_courier = Column(String(50), nullable=False, server_default=text("''"))
    pkwid_declaredValue = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))


class KqsPlOperacje(Base):
    __tablename__ = 'kqs_pl_operacje'

    numer = Column(BigInteger, primary_key=True)
    klient_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    punkty = Column(BigInteger, nullable=False, server_default=text("'0'"))
    komentarz = Column(Text, nullable=False)
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsPlatformaratalna(Base):
    __tablename__ = 'kqs_platformaratalna'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, unique=True, server_default=text("'0'"))
    identyfikator = Column(String(50), nullable=False, server_default=text("''"))
    status = Column(String(50), nullable=False, server_default=text("''"))


class KqsPlatnosciDotpay(Base):
    __tablename__ = 'kqs_platnosci_dotpay'

    numer = Column(BigInteger, primary_key=True)
    dotpay_status = Column(Integer, nullable=False, server_default=text("'0'"))
    dotpay_control = Column(String(50), nullable=False, server_default=text("''"))
    dotpay_tid = Column(String(30), nullable=False, server_default=text("''"))
    dotpay_amount = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    dotpay_waluta = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    dotpay_paid = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    dotpay_channel = Column(String(10), nullable=False, server_default=text("''"))
    dotpay_description = Column(String(255), nullable=False, server_default=text("''"))
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsPlatnosciMoneybooker(Base):
    __tablename__ = 'kqs_platnosci_moneybookers'

    numer = Column(BigInteger, primary_key=True)
    mb_session_id = Column(String(64), nullable=False, server_default=text("''"))
    mb_order_id = Column(String(25), nullable=False, server_default=text("''"))
    mb_kwota = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    mb_waluta = Column(String(15), nullable=False, server_default=text("''"))
    mb_status = Column(Integer, nullable=False, server_default=text("'0'"))
    mb_raport = Column(Text, nullable=False)
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsPlatnosciP24(Base):
    __tablename__ = 'kqs_platnosci_p24'

    numer = Column(BigInteger, primary_key=True)
    p24_session_id = Column(String(64), nullable=False, server_default=text("''"))
    p24_order_id = Column(Integer, nullable=False, server_default=text("'0'"))
    p24_order_id_full = Column(String(32), nullable=False, server_default=text("''"))
    p24_kwota = Column(BigInteger, nullable=False, server_default=text("'0'"))
    p24_status = Column(Integer, nullable=False, server_default=text("'0'"))
    p24_raport = Column(Text, nullable=False)
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsPlatnosciPl(Base):
    __tablename__ = 'kqs_platnosci_pl'

    numer = Column(BigInteger, primary_key=True)
    pl_session_id = Column(String(64), nullable=False, server_default=text("''"))
    pl_order_id = Column(Integer, nullable=False, server_default=text("'0'"))
    pl_kwota = Column(BigInteger, nullable=False, server_default=text("'0'"))
    pl_status = Column(Integer, nullable=False, server_default=text("'0'"))
    pl_raport = Column(Text, nullable=False)
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsPliki(Base):
    __tablename__ = 'kqs_pliki'

    numer = Column(BigInteger, primary_key=True)
    produkt_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    nazwa = Column(String(255), nullable=False, server_default=text("''"))
    opis = Column(Text, nullable=False)
    plik = Column(String(255), nullable=False, server_default=text("''"))
    rozmiar = Column(String(25), nullable=False, server_default=text("''"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    licznik = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsPlikiPobrania(Base):
    __tablename__ = 'kqs_pliki_pobrania'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    plik_numer = Column(BigInteger, nullable=False, server_default=text("'0'"))
    plik_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    referent = Column(String(255), nullable=False, server_default=text("''"))
    data_pobrania = Column(BigInteger, nullable=False, server_default=text("'0'"))
    adres_ip = Column(String(50), nullable=False, server_default=text("''"))


class KqsPlikiSprzedaz(Base):
    __tablename__ = 'kqs_pliki_sprzedaz'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(200), nullable=False, server_default=text("''"))
    plik_nazwa_tmp = Column(String(100), nullable=False, server_default=text("''"))
    plik_nazwa_org = Column(String(100), nullable=False, server_default=text("''"))
    plik_typ = Column(String(100), nullable=False, server_default=text("''"))
    plik_rozmiar = Column(BigInteger, nullable=False, server_default=text("'0'"))
    katalog = Column(String(100), nullable=False, server_default=text("''"))
    opis = Column(Text, nullable=False)


class KqsPocztaWychodzaca(Base):
    __tablename__ = 'kqs_poczta_wychodzaca'

    numer = Column(BigInteger, primary_key=True)
    m_subject = Column(String(255), nullable=False, server_default=text("''"))
    m_sender = Column(String(255), nullable=False, server_default=text("''"))
    m_recipient = Column(String(255), nullable=False, server_default=text("''"))
    m_reply_to = Column(String(255), nullable=False, server_default=text("''"))
    m_date = Column(String(11), nullable=False, server_default=text("''"))
    status = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    raport = Column(String(255), nullable=False, server_default=text("''"))


class KqsPorownywarka(Base):
    __tablename__ = 'kqs_porownywarka'

    numer = Column(BigInteger, primary_key=True)
    produkt_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    klient_id = Column(String(50), nullable=False, index=True, server_default=text("''"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsPpKody(Base):
    __tablename__ = 'kqs_pp_kody'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(255), nullable=False, server_default=text("''"))
    tresc_kodu = Column(Text, nullable=False)


class KqsPpKonfiguracja(Base):
    __tablename__ = 'kqs_pp_konfiguracja'

    numer = Column(BigInteger, primary_key=True)
    aktywnosc = Column(Integer, nullable=False, server_default=text("'0'"))
    prowizja = Column(Float(4, True), nullable=False, server_default=text("'0.00'"))
    limit_wyplata = Column(BigInteger, nullable=False, server_default=text("'0'"))
    regulamin = Column(Text, nullable=False)
    limit_dni = Column(BigInteger, nullable=False, server_default=text("'0'"))
    dane_faktury = Column(Text, nullable=False)
    dane_wysylki = Column(Text, nullable=False)
    dowolny_referent = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsPpProwizje(Base):
    __tablename__ = 'kqs_pp_prowizje'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    partner_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    wartosc_prowizji = Column(Float(4, True), nullable=False, server_default=text("'0.00'"))
    wartosc_pieniezna = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsPpRuch(Base):
    __tablename__ = 'kqs_pp_ruch'

    numer = Column(BigInteger, primary_key=True)
    adres_referenta = Column(String(255), nullable=False, server_default=text("''"))
    partner_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    ip_klienta = Column(String(25), nullable=False, server_default=text("''"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsPpUzytkownicy(Base):
    __tablename__ = 'kqs_pp_uzytkownicy'

    numer = Column(BigInteger, primary_key=True)
    uid = Column(String(100), nullable=False, server_default=text("''"))
    uzytkownik_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    adres_strony = Column(String(100), nullable=False, server_default=text("''"))
    imie_firma = Column(String(100), nullable=False, server_default=text("''"))
    nazwisko_nip = Column(String(100), nullable=False, server_default=text("''"))
    ulica_dom = Column(String(100), nullable=False, server_default=text("''"))
    kod_pocztowy = Column(String(10), nullable=False, server_default=text("''"))
    miasto = Column(String(50), nullable=False, server_default=text("''"))
    email = Column(String(100), nullable=False, server_default=text("''"))
    telefon = Column(String(25), nullable=False, server_default=text("''"))
    numer_konta = Column(String(30), nullable=False, server_default=text("''"))
    aktywny = Column(Integer, nullable=False, server_default=text("'0'"))
    data_przystapienia = Column(BigInteger, nullable=False, server_default=text("'0'"))
    prowizja = Column(Float(4, True), nullable=False, server_default=text("'0.00'"))


class KqsPpWyplaty(Base):
    __tablename__ = 'kqs_pp_wyplaty'

    numer = Column(BigInteger, primary_key=True)
    partner_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    wartosc = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    rodzaj = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    status = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsPpZakupy(Base):
    __tablename__ = 'kqs_pp_zakupy'

    numer = Column(BigInteger, primary_key=True)
    ip_klienta = Column(String(25), nullable=False, server_default=text("''"))
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    partner_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    status = Column(Integer, nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsProducenci(Base):
    __tablename__ = 'kqs_producenci'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(100), nullable=False, unique=True, server_default=text("''"))
    wizytowka = Column(Text, nullable=False)
    logo_producenta = Column(String(100), nullable=False, server_default=text("''"))
    wizytowka_2 = Column(Text, nullable=False)


class KqsProdukty(Base):
    __tablename__ = 'kqs_produkty'
    __table_args__ = (
        Index('szukaj_opis', 'nazwa', 'krotki_opis', 'opis', 'opis_2', 'opis_3'),
    )

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(200), nullable=False, index=True, server_default=text("''"))
    cena = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    cena_prom = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    opis = Column(Text, nullable=False)
    promocja = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    user_id = Column(Integer, nullable=False, server_default=text("'1'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    producent_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    aktywne = Column(Integer, nullable=False, server_default=text("'0'"))
    stan = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    ogladane = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    kupione = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    waga = Column(Float(8, True), nullable=False, server_default=text("'0.00'"))
    podatek = Column(Integer, nullable=False, server_default=text("'0'"))
    strona_glowna = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    dostepnosc = Column(BigInteger, nullable=False, server_default=text("'0'"))
    meta_keywords = Column(String(255), nullable=False, server_default=text("''"))
    meta_description = Column(String(255), nullable=False, server_default=text("''"))
    krotki_opis = Column(Text, nullable=False)
    PKWiU = Column(String(25), nullable=False, server_default=text("''"))
    jm = Column(String(10), nullable=False, server_default=text("''"))
    cena_hurt = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    cena_hurt_prom = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    promocja_hurt = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    kod_produktu = Column(String(50), nullable=False, server_default=text("''"))
    dostawca_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    gradacja_detal = Column(String(255), nullable=False, server_default=text("''"))
    gradacja_hurt = Column(String(255), nullable=False, server_default=text("''"))
    kod_u_dostawcy = Column(String(50), nullable=False, server_default=text("''"))
    cecha_01 = Column(Text, nullable=False)
    cecha_02 = Column(String(255), nullable=False, server_default=text("''"))
    cecha_03 = Column(String(255), nullable=False, server_default=text("''"))
    cecha_04 = Column(String(255), nullable=False, server_default=text("''"))
    cecha_05 = Column(String(255), nullable=False, server_default=text("''"))
    cecha_06 = Column(String(255), nullable=False, server_default=text("''"))
    cecha_07 = Column(String(255), nullable=False, server_default=text("''"))
    cecha_08 = Column(String(255), nullable=False, server_default=text("''"))
    cecha_09 = Column(String(255), nullable=False, server_default=text("''"))
    cecha_10 = Column(String(255), nullable=False, server_default=text("''"))
    oznaczenie_nowosc = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    dostepnosc_po_wyczerpaniu = Column(BigInteger, nullable=False, server_default=text("'0'"))
    wylacz_atrybuty_magazyn = Column(Integer, nullable=False, server_default=text("'0'"))
    waluta_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    cena_03 = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    cena_prom_03 = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    promocja_03 = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    gradacja_03 = Column(String(255), nullable=False, server_default=text("''"))
    cena_04 = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    cena_prom_04 = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    promocja_04 = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    gradacja_04 = Column(String(255), nullable=False, server_default=text("''"))
    cena_05 = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    cena_prom_05 = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    promocja_05 = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    gradacja_05 = Column(String(255), nullable=False, server_default=text("''"))
    cena_06 = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    cena_prom_06 = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    promocja_06 = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    gradacja_06 = Column(String(255), nullable=False, server_default=text("''"))
    cena_07 = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    cena_prom_07 = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    promocja_07 = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    gradacja_07 = Column(String(255), nullable=False, server_default=text("''"))
    cena_08 = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    cena_prom_08 = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    promocja_08 = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    gradacja_08 = Column(String(255), nullable=False, server_default=text("''"))
    cena_09 = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    cena_prom_09 = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    promocja_09 = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    gradacja_09 = Column(String(255), nullable=False, server_default=text("''"))
    cena_10 = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    cena_prom_10 = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    promocja_10 = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    gradacja_10 = Column(String(255), nullable=False, server_default=text("''"))
    kod_plu = Column(String(25), nullable=False, server_default=text("''"))
    kod_kreskowy = Column(String(25), nullable=False, server_default=text("''"))
    gab_wysokosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    gab_szerokosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    gab_dlugosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    wybrany = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    wyprzedaz = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    bestseller = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    kod_u_producenta = Column(String(50), nullable=False, server_default=text("''"))
    gwarancja = Column(String(25), nullable=False, server_default=text("''"))
    archiwalny = Column(Integer, nullable=False, server_default=text("'0'"))
    cechy_grupa = Column(BigInteger, nullable=False, server_default=text("'0'"))
    opis_2 = Column(Text, nullable=False)
    opis_3 = Column(Text, nullable=False)
    porownywarki = Column(Integer, nullable=False, server_default=text("'0'"))
    paczka = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    cena_zakupu = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    backorder = Column(Integer, nullable=False, server_default=text("'0'"))
    bezplatna_dostawa = Column(Integer, nullable=False, index=True, server_default=text("'0'"))


class KqsProduktyAtrybuty(Base):
    __tablename__ = 'kqs_produkty_atrybuty'
    __table_args__ = (
        Index('opcja_wartosc', 'opcja_id', 'wartosc_id'),
    )

    numer = Column(BigInteger, primary_key=True)
    produkt_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    opcja_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    wartosc_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    zmiana_ceny = Column(Integer, nullable=False, server_default=text("'0'"))
    zmiana_wartosc = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    zmiana_jednostka = Column(Integer, nullable=False, server_default=text("'0'"))
    zmiana_info = Column(Integer, nullable=False, server_default=text("'0'"))
    widocznosc = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsProduktyAtrybutyMagazyn(Base):
    __tablename__ = 'kqs_produkty_atrybuty_magazyn'

    numer = Column(BigInteger, primary_key=True)
    pakiet_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    atrybut_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))


class KqsProduktyAtrybutyMagazynPakiety(Base):
    __tablename__ = 'kqs_produkty_atrybuty_magazyn_pakiety'

    numer = Column(BigInteger, primary_key=True)
    produkt_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    stan = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    dostepnosc_stala = Column(BigInteger, nullable=False, server_default=text("'0'"))
    dostepnosc_po_wyczerpaniu = Column(BigInteger, nullable=False, server_default=text("'0'"))
    kod_produktu = Column(String(50), nullable=False, server_default=text("''"))
    kod_producenta = Column(String(50), nullable=False, server_default=text("''"))
    kod_dostawcy = Column(String(50), nullable=False, server_default=text("''"))
    kod_kreskowy = Column(String(25), nullable=False, server_default=text("''"))
    kod_plu = Column(String(25), nullable=False, server_default=text("''"))
    waga = Column(Float(8, True), nullable=False, server_default=text("'0.00'"))


class KqsProduktyBkp(Base):
    __tablename__ = 'kqs_produkty_bkp'
    __table_args__ = (
        Index('szukaj_opis', 'nazwa', 'opis', 'krotki_opis', 'kod_produktu'),
        Index('szukaj_nazwa', 'nazwa', 'kod_produktu')
    )

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(200), nullable=False, server_default=text("''"))
    cena = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    cena_prom = Column(Float(asdecimal=True), nullable=False, index=True, server_default=text("'0'"))
    opis = Column(Text, nullable=False)
    promocja = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    user_id = Column(Integer, nullable=False, server_default=text("'1'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    producent_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    aktywne = Column(Integer, nullable=False, server_default=text("'0'"))
    stan = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    ogladane = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    kupione = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    waga = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    podatek = Column(Integer, nullable=False, server_default=text("'0'"))
    strona_glowna = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    dostepnosc = Column(BigInteger, nullable=False, server_default=text("'0'"))
    meta_keywords = Column(String(255), nullable=False, server_default=text("''"))
    meta_description = Column(String(255), nullable=False, server_default=text("''"))
    krotki_opis = Column(Text, nullable=False)
    PKWiU = Column(String(25), nullable=False, server_default=text("''"))
    jm = Column(String(10), nullable=False, server_default=text("''"))
    cena_hurt = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    cena_hurt_prom = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    promocja_hurt = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    kod_produktu = Column(String(50), nullable=False, server_default=text("''"))
    dostawca_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    gradacja_detal = Column(String(255), nullable=False, server_default=text("''"))
    gradacja_hurt = Column(String(255), nullable=False, server_default=text("''"))
    opis_html = Column(Integer, nullable=False, server_default=text("'0'"))
    kod_u_dostawcy = Column(String(50), nullable=False, server_default=text("''"))
    cecha_01 = Column(Text, nullable=False)
    cecha_02 = Column(String(255), nullable=False, server_default=text("''"))
    cecha_03 = Column(String(255), nullable=False, server_default=text("''"))
    cecha_04 = Column(String(255), nullable=False, server_default=text("''"))
    cecha_05 = Column(String(255), nullable=False, server_default=text("''"))
    cecha_06 = Column(String(255), nullable=False, server_default=text("''"))
    cecha_07 = Column(String(255), nullable=False, server_default=text("''"))
    cecha_08 = Column(String(255), nullable=False, server_default=text("''"))
    cecha_09 = Column(String(255), nullable=False, server_default=text("''"))
    cecha_10 = Column(String(255), nullable=False, server_default=text("''"))
    atrybuty_magazyn = Column(Text, nullable=False)
    oznaczenie_nowosc = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    dostepnosc_po_wyczerpaniu = Column(BigInteger, nullable=False, server_default=text("'0'"))
    wylacz_atrybuty_magazyn = Column(Integer, nullable=False, server_default=text("'0'"))
    waluta_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    cena_03 = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    cena_prom_03 = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    promocja_03 = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    gradacja_03 = Column(String(255), nullable=False, server_default=text("''"))
    cena_04 = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    cena_prom_04 = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    promocja_04 = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    gradacja_04 = Column(String(255), nullable=False, server_default=text("''"))
    cena_05 = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    cena_prom_05 = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    promocja_05 = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    gradacja_05 = Column(String(255), nullable=False, server_default=text("''"))
    cena_06 = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    cena_prom_06 = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    promocja_06 = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    gradacja_06 = Column(String(255), nullable=False, server_default=text("''"))
    cena_07 = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    cena_prom_07 = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    promocja_07 = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    gradacja_07 = Column(String(255), nullable=False, server_default=text("''"))
    cena_08 = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    cena_prom_08 = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    promocja_08 = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    gradacja_08 = Column(String(255), nullable=False, server_default=text("''"))
    cena_09 = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    cena_prom_09 = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    promocja_09 = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    gradacja_09 = Column(String(255), nullable=False, server_default=text("''"))
    cena_10 = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    cena_prom_10 = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    promocja_10 = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    gradacja_10 = Column(String(255), nullable=False, server_default=text("''"))
    kod_plu = Column(String(25), nullable=False, server_default=text("''"))
    kod_kreskowy = Column(String(25), nullable=False, server_default=text("''"))
    gab_wysokosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    gab_szerokosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    gab_dlugosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    wybrany = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    wyprzedaz = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    bestseller = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    kod_u_producenta = Column(String(50), nullable=False, server_default=text("''"))
    gwarancja = Column(String(25), nullable=False, server_default=text("''"))
    archiwalny = Column(Integer, nullable=False, server_default=text("'0'"))
    cechy_grupa = Column(BigInteger, nullable=False, server_default=text("'0'"))
    opis_2 = Column(Text, nullable=False)
    opis_3 = Column(Text, nullable=False)


class KqsProduktyCechy(Base):
    __tablename__ = 'kqs_produkty_cechy'
    __table_args__ = (
        Index('opcja_wartosc', 'opcja_id', 'wartosc_id'),
    )

    numer = Column(BigInteger, primary_key=True)
    produkt_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    opcja_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    wartosc_id = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsProduktyCechyOpcje(Base):
    __tablename__ = 'kqs_produkty_cechy_opcje'

    numer = Column(BigInteger, primary_key=True)
    opcja = Column(String(255), nullable=False, server_default=text("''"))
    wyszukiwarka = Column(Integer, nullable=False, server_default=text("'0'"))
    kolejnosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    opis = Column(Text, nullable=False)
    rodzaj = Column(Integer, nullable=False, server_default=text("'0'"))
    wartosci_liczbowe = Column(Integer, nullable=False, server_default=text("'0'"))
    katalog = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsProduktyCechyWartosci(Base):
    __tablename__ = 'kqs_produkty_cechy_wartosci'

    numer = Column(BigInteger, primary_key=True)
    wartosc = Column(Text, nullable=False)
    kolejnosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))


class KqsProduktyGrati(Base):
    __tablename__ = 'kqs_produkty_gratis'

    numer = Column(BigInteger, primary_key=True)
    produkt_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    minimalne_zakupy = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))


t_kqs_produkty_kategorie = Table(
    'kqs_produkty_kategorie', metadata,
    Column('produkt_id', BigInteger, nullable=False, index=True, server_default=text("'0'")),
    Column('kategoria_id', BigInteger, nullable=False, index=True, server_default=text("'0'"))
)


class KqsProduktyOpcje(Base):
    __tablename__ = 'kqs_produkty_opcje'

    numer = Column(BigInteger, primary_key=True)
    opcja = Column(String(255), nullable=False, server_default=text("''"))
    opis = Column(String(255), nullable=False, server_default=text("''"))
    kolejnosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    magazyn = Column(Integer, nullable=False, server_default=text("'0'"))
    format = Column(Integer, nullable=False, server_default=text("'0'"))
    rodzaj = Column(Integer, nullable=False, server_default=text("'0'"))
    wartosci_liczbowe = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsProduktyPliki(Base):
    __tablename__ = 'kqs_produkty_pliki'

    numer = Column(BigInteger, primary_key=True)
    produkt_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    plik_id = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsProduktyPokrewne(Base):
    __tablename__ = 'kqs_produkty_pokrewne'

    numer = Column(BigInteger, primary_key=True)
    produkt_id1 = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt_id2 = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))


class KqsProduktyWartosci(Base):
    __tablename__ = 'kqs_produkty_wartosci'

    numer = Column(BigInteger, primary_key=True)
    wartosc = Column(String(255), nullable=False, index=True, server_default=text("''"))
    kolejnosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    zmiana_ceny = Column(Integer, nullable=False, server_default=text("'0'"))
    zmiana_wartosc = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    zmiana_jednostka = Column(Integer, nullable=False, server_default=text("'0'"))
    produkt_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    opis = Column(String(255), nullable=False, server_default=text("''"))


t_kqs_produkty_wersje = Table(
    'kqs_produkty_wersje', metadata,
    Column('produkt_id', BigInteger, nullable=False, index=True, server_default=text("'0'")),
    Column('wersja_id', BigInteger, nullable=False, index=True, server_default=text("'0'"))
)


class KqsPrzechowalnia(Base):
    __tablename__ = 'kqs_przechowalnia'

    numer = Column(BigInteger, primary_key=True)
    produkt_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    klient_id = Column(String(50), nullable=False, index=True, server_default=text("''"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsPrzesylki(Base):
    __tablename__ = 'kqs_przesylki'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(100), nullable=False, server_default=text("''"))
    opis = Column(Text, nullable=False)
    koszt = Column(Float(10, True), nullable=False, server_default=text("'0.00'"))
    dotpay = Column(Integer, nullable=False, server_default=text("'0'"))
    max_waga = Column(Float(8, True), nullable=False, server_default=text("'0.00'"))
    min_waga = Column(Float(8, True), nullable=False, server_default=text("'0.00'"))
    p24 = Column(Integer, nullable=False, server_default=text("'0'"))
    kwota_free = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    payu = Column(Integer, nullable=False, server_default=text("'0'"))
    paypal = Column(Integer, nullable=False, server_default=text("'0'"))
    moneybookers = Column(Integer, nullable=False, server_default=text("'0'"))
    zaplata_przelew = Column(Integer, nullable=False, server_default=text("'0'"))
    zaplata_gotowka = Column(Integer, nullable=False, server_default=text("'0'"))
    pobranie_procenty = Column(Float(4, True), nullable=False, server_default=text("'0.00'"))
    pobranie_kwota = Column(Float(8, True), nullable=False, server_default=text("'0.00'"))
    kwota_free_pobranie = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    eraty = Column(Integer, nullable=False, server_default=text("'0'"))
    max_zamowienie = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    min_zamowienie = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    inpost_aktywacja = Column(Integer, nullable=False, server_default=text("'0'"))
    inpost_rozmiar = Column(Integer, nullable=False, server_default=text("'0'"))
    inpost_ubezpieczenie = Column(Integer, nullable=False, server_default=text("'0'"))
    opis_gotowka = Column(Text, nullable=False)
    tpay = Column(Integer, nullable=False, server_default=text("'0'"))
    przesylka_specjalna = Column(Integer, nullable=False, server_default=text("'0'"))
    mbank_raty = Column(Integer, nullable=False, server_default=text("'0'"))
    kolejnosc = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    paybynet = Column(Integer, nullable=False, server_default=text("'0'"))
    platformaratalna = Column(Integer, nullable=False, server_default=text("'0'"))
    sofort = Column(Integer, nullable=False, server_default=text("'0'"))
    ferbuy = Column(Integer, nullable=False, server_default=text("'0'"))
    poziom_cenowy = Column(Integer, nullable=False, server_default=text("'0'"))
    leasing = Column(Integer, nullable=False, server_default=text("'0'"))
    caraty = Column(Integer, nullable=False, server_default=text("'0'"))
    kraj_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    bluemedia = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsRabaty(Base):
    __tablename__ = 'kqs_rabaty'

    numer = Column(BigInteger, primary_key=True)
    klient = Column(BigInteger, nullable=False, server_default=text("'0'"))
    typ = Column(Integer, nullable=False, server_default=text("'0'"))
    minimum = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    wartosc = Column(Float(4, True), nullable=False, server_default=text("'0.00'"))
    kategoria_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data_wygasniecia = Column(BigInteger, nullable=False, server_default=text("'0'"))
    producent_id = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsRabatyAuto(Base):
    __tablename__ = 'kqs_rabaty_auto'

    numer = Column(BigInteger, primary_key=True)
    kwota = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    rabat = Column(Float(4, True), nullable=False, server_default=text("'0.00'"))
    typ_konta = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsRachunki(Base):
    __tablename__ = 'kqs_rachunki'

    numer = Column(BigInteger, primary_key=True)
    nr_dokumentu = Column(BigInteger, nullable=False, server_default=text("'0'"))
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    data_wystawienia = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data_sprzedazy = Column(BigInteger, nullable=False, server_default=text("'0'"))
    forma_zaplaty = Column(String(50), nullable=False, server_default=text("''"))
    termin_zaplaty = Column(String(25), nullable=False, server_default=text("''"))
    miejsce_wystawienia = Column(String(50), nullable=False, server_default=text("''"))
    administrator_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    administrator_nazwa = Column(String(100), nullable=False, server_default=text("''"))
    data_duplikatu = Column(BigInteger, nullable=False, server_default=text("'0'"))
    adnotacje = Column(Text, nullable=False)
    fraza_podsumowujaca = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsRecenzje(Base):
    __tablename__ = 'kqs_recenzje'

    numer = Column(BigInteger, primary_key=True)
    produkt_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    ocena = Column(Integer, nullable=False, server_default=text("'0'"))
    recenzja = Column(Text, nullable=False)
    autor = Column(String(30), nullable=False, server_default=text("''"))
    autor_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    adres_ip = Column(String(25), nullable=False, server_default=text("''"))
    weryfikacja = Column(Integer, nullable=False, server_default=text("'0'"))
    adnotacja = Column(Text, nullable=False)


class KqsRejestracja(Base):
    __tablename__ = 'kqs_rejestracja'

    user_id = Column(BigInteger, primary_key=True)
    email = Column(String(200), nullable=False, unique=True, server_default=text("''"))
    haslo = Column(String(100), nullable=False, server_default=text("''"))
    imie = Column(String(50), nullable=False, server_default=text("''"))
    nazwisko = Column(String(100), nullable=False, server_default=text("''"))
    ulica = Column(String(100), nullable=False, server_default=text("''"))
    dom = Column(String(20), nullable=False, server_default=text("''"))
    kod = Column(String(10), nullable=False, server_default=text("''"))
    miasto = Column(String(100), nullable=False, server_default=text("''"))
    wojewodztwo = Column(String(100), nullable=False, server_default=text("''"))
    telefon = Column(String(20), nullable=False, server_default=text("''"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    uuid = Column(String(100), nullable=False, server_default=text("''"))
    aktywne = Column(String(1), nullable=False, server_default=text("'0'"))
    firma = Column(String(255), nullable=False, server_default=text("''"))
    nip = Column(String(100), nullable=False, server_default=text("''"))
    f_ulica = Column(String(100), nullable=False, server_default=text("''"))
    f_dom = Column(String(20), nullable=False, server_default=text("''"))
    f_kod = Column(String(10), nullable=False, server_default=text("''"))
    f_miasto = Column(String(100), nullable=False, server_default=text("''"))
    f_woj = Column(String(100), nullable=False, server_default=text("''"))
    f_tel = Column(String(20), nullable=False, server_default=text("''"))
    nowosci = Column(Integer, nullable=False, server_default=text("'0'"))
    hurtownik = Column(Integer, nullable=False, server_default=text("'0'"))
    kraj = Column(String(100), nullable=False, server_default=text("''"))
    f_kraj = Column(String(100), nullable=False, server_default=text("''"))
    m_limit = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    grupa_rabatowa = Column(BigInteger, nullable=False, server_default=text("'0'"))
    termin_przelewu = Column(Integer, nullable=False, server_default=text("'0'"))
    regon = Column(String(9), nullable=False, server_default=text("''"))
    kontakt_gg = Column(String(12), nullable=False, server_default=text("''"))
    kontakt_skype = Column(String(35), nullable=False, server_default=text("''"))
    strona_www = Column(String(100), nullable=False, server_default=text("''"))
    branza_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    admin_obslugujacy = Column(BigInteger, nullable=False, server_default=text("'0'"))
    ostatnie_logowanie = Column(BigInteger, nullable=False, server_default=text("'0'"))
    nip_unikalny = Column(String(50), nullable=False, server_default=text("''"))
    limit_kredytowy = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    vat_zero = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsRejestracjaAdresy(Base):
    __tablename__ = 'kqs_rejestracja_adresy'

    numer = Column(BigInteger, primary_key=True)
    klient_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    identyfikator = Column(String(50), nullable=False, server_default=text("''"))
    odbiorca = Column(String(100), nullable=False, server_default=text("''"))
    ulica = Column(String(100), nullable=False, server_default=text("''"))
    dom = Column(String(20), nullable=False, server_default=text("''"))
    kod = Column(String(10), nullable=False, server_default=text("''"))
    miasto = Column(String(100), nullable=False, server_default=text("''"))


class KqsRejestracjaBranze(Base):
    __tablename__ = 'kqs_rejestracja_branze'

    numer = Column(BigInteger, primary_key=True)
    branza = Column(String(50), nullable=False, server_default=text("''"))
    poziom_cenowy = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsRejestracjaPliki(Base):
    __tablename__ = 'kqs_rejestracja_pliki'

    numer = Column(BigInteger, primary_key=True)
    klient_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    nazwa = Column(String(200), nullable=False, server_default=text("''"))
    plik_nazwa_tmp = Column(String(100), nullable=False, server_default=text("''"))
    plik_nazwa_org = Column(String(100), nullable=False, server_default=text("''"))
    plik_typ = Column(String(100), nullable=False, server_default=text("''"))
    plik_rozmiar = Column(BigInteger, nullable=False, server_default=text("'0'"))
    katalog = Column(String(100), nullable=False, server_default=text("''"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    opis = Column(Text, nullable=False)


class KqsRejestracjaPrzedstawiciele(Base):
    __tablename__ = 'kqs_rejestracja_przedstawiciele'

    numer = Column(BigInteger, primary_key=True)
    klient_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    identyfikator = Column(String(50), nullable=False, server_default=text("''"))
    imie = Column(String(50), nullable=False, server_default=text("''"))
    nazwisko = Column(String(100), nullable=False, server_default=text("''"))
    email = Column(String(100), nullable=False, server_default=text("''"))
    haslo = Column(String(100), nullable=False, server_default=text("''"))
    telefon = Column(String(20), nullable=False, server_default=text("''"))
    funkcja = Column(String(100), nullable=False, server_default=text("''"))
    opis = Column(Text, nullable=False)
    data_dodania = Column(BigInteger, nullable=False, server_default=text("'0'"))
    ostatnie_logowanie = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsSamModele(Base):
    __tablename__ = 'kqs_sam_modele'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(100), nullable=False, server_default=text("''"))
    marka_id = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsSamWersje(Base):
    __tablename__ = 'kqs_sam_wersje'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(100), nullable=False, server_default=text("''"))
    model_id = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsSendit(Base):
    __tablename__ = 'kqs_sendit'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    sendit_order_nr = Column(String(30), nullable=False, server_default=text("''"))
    sendit_status = Column(String(255), nullable=False, server_default=text("''"))
    sendit_status_nr = Column(Integer, nullable=False, server_default=text("'0'"))
    sendit_lpnumber = Column(String(100), nullable=False, server_default=text("''"))
    sendit_tracking_code = Column(String(255), nullable=False, server_default=text("''"))
    sendit_protocol_number = Column(String(40), nullable=False, server_default=text("''"))
    sendit_cod = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    sendit_brutto = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    sendit_courier = Column(String(45), nullable=False, server_default=text("''"))


class KqsSmsapi(Base):
    __tablename__ = 'kqs_smsapi'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    sms_id = Column(String(25), nullable=False, server_default=text("''"))
    odbiorca = Column(String(50), nullable=False, server_default=text("''"))
    wiadomosc = Column(Text, nullable=False)
    status = Column(String(5), nullable=False, server_default=text("''"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsSofort(Base):
    __tablename__ = 'kqs_sofort'

    numer = Column(BigInteger, primary_key=True)
    sofort_amount = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    sofort_amount_received = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    sofort_currency = Column(String(5), nullable=False, server_default=text("''"))
    sofort_currency_received = Column(String(5), nullable=False, server_default=text("''"))
    sofort_transaction_id = Column(String(50), nullable=False, server_default=text("''"))
    sofort_transaction_url = Column(String(50), nullable=False, server_default=text("''"))
    sofort_status = Column(String(15), nullable=False, server_default=text("''"))
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsSondy(Base):
    __tablename__ = 'kqs_sondy'

    numer = Column(BigInteger, primary_key=True)
    sonda = Column(String(255), nullable=False, server_default=text("''"))
    odp_01 = Column(String(255), nullable=False, server_default=text("''"))
    odp_02 = Column(String(255), nullable=False, server_default=text("''"))
    odp_03 = Column(String(255), nullable=False, server_default=text("''"))
    odp_04 = Column(String(255), nullable=False, server_default=text("''"))
    odp_05 = Column(String(255), nullable=False, server_default=text("''"))
    odp_06 = Column(String(255), nullable=False, server_default=text("''"))
    odp_07 = Column(String(255), nullable=False, server_default=text("''"))
    odp_08 = Column(String(255), nullable=False, server_default=text("''"))
    odp_09 = Column(String(255), nullable=False, server_default=text("''"))
    odp_10 = Column(String(255), nullable=False, server_default=text("''"))
    odp_11 = Column(String(255), nullable=False, server_default=text("''"))
    odp_12 = Column(String(255), nullable=False, server_default=text("''"))
    odp_13 = Column(String(255), nullable=False, server_default=text("''"))
    odp_14 = Column(String(255), nullable=False, server_default=text("''"))
    odp_15 = Column(String(255), nullable=False, server_default=text("''"))
    odp_16 = Column(String(255), nullable=False, server_default=text("''"))
    odp_17 = Column(String(255), nullable=False, server_default=text("''"))
    odp_18 = Column(String(255), nullable=False, server_default=text("''"))
    odp_19 = Column(String(255), nullable=False, server_default=text("''"))
    odp_20 = Column(String(255), nullable=False, server_default=text("''"))
    odp_votes_01 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_02 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_03 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_04 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_05 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_06 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_07 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_08 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_09 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_10 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_11 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_12 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_13 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_14 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_15 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_16 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_17 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_18 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_19 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    odp_votes_20 = Column(BigInteger, nullable=False, server_default=text("'0'"))
    total_votes = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data_rozpoczecia = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data_zakonczenia = Column(BigInteger, nullable=False, server_default=text("'0'"))
    status = Column(String(11), nullable=False, server_default=text("''"))


t_kqs_sondy_glosy = Table(
    'kqs_sondy_glosy', metadata,
    Column('sonda_id', BigInteger, nullable=False, server_default=text("'0'")),
    Column('adres_ip', String(50), nullable=False, index=True, server_default=text("''")),
    Column('adres_host', String(50), nullable=False, index=True, server_default=text("''")),
    Column('data', BigInteger, nullable=False, server_default=text("'0'"))
)


class KqsStatusyDodatkowe(Base):
    __tablename__ = 'kqs_statusy_dodatkowe'

    numer = Column(BigInteger, primary_key=True)
    status_nazwa = Column(String(100), nullable=False, server_default=text("''"))
    kolejnosc = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsSubskrypcja(Base):
    __tablename__ = 'kqs_subskrypcja'

    numer = Column(BigInteger, primary_key=True)
    adres_email = Column(String(100), nullable=False, server_default=text("''"))
    aktywne = Column(Integer, nullable=False, server_default=text("'0'"))
    uid = Column(String(100), nullable=False, server_default=text("''"))
    grupa_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    ip_rejestracja = Column(String(20), nullable=False, server_default=text("''"))
    ip_aktywacja = Column(String(20), nullable=False, server_default=text("''"))
    data_rejestracja = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data_aktywacja = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsSubskrypcjaGrupy(Base):
    __tablename__ = 'kqs_subskrypcja_grupy'

    numer = Column(BigInteger, primary_key=True)
    grupa = Column(String(25), nullable=False, server_default=text("''"))
    niewidocznosc = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsTransferujPl(Base):
    __tablename__ = 'kqs_transferuj_pl'

    numer = Column(BigInteger, primary_key=True)
    tr_crc = Column(String(128), nullable=False, server_default=text("''"))
    tr_id = Column(String(128), nullable=False, server_default=text("''"))
    tr_amount = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    tr_paid = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    tr_status = Column(Integer, nullable=False, server_default=text("'0'"))
    tr_raport = Column(Text, nullable=False)
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsUser(Base):
    __tablename__ = 'kqs_users'

    numer = Column(BigInteger, primary_key=True)
    login = Column(String(25), nullable=False, unique=True, server_default=text("''"))
    haslo = Column(String(50), nullable=False, server_default=text("''"))
    imie = Column(String(50), nullable=False, server_default=text("''"))
    nazwisko = Column(String(50), nullable=False, server_default=text("''"))
    mail = Column(String(100), nullable=False, server_default=text("''"))
    rodzaj = Column(Integer, nullable=False, server_default=text("'1'"))
    kat = Column(Integer, nullable=False, server_default=text("'0'"))
    kat_add = Column(Integer, nullable=False, server_default=text("'0'"))
    kat_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    kat_del = Column(Integer, nullable=False, server_default=text("'0'"))
    pro = Column(Integer, nullable=False, server_default=text("'0'"))
    pro_add = Column(Integer, nullable=False, server_default=text("'0'"))
    pro_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    pro_del = Column(Integer, nullable=False, server_default=text("'0'"))
    prod = Column(Integer, nullable=False, server_default=text("'0'"))
    prod_add = Column(Integer, nullable=False, server_default=text("'0'"))
    prod_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    prod_del = Column(Integer, nullable=False, server_default=text("'0'"))
    gal = Column(Integer, nullable=False, server_default=text("'0'"))
    gal_add = Column(Integer, nullable=False, server_default=text("'0'"))
    gal_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    gal_del = Column(Integer, nullable=False, server_default=text("'0'"))
    klienci = Column(Integer, nullable=False, server_default=text("'0'"))
    klienci_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    klienci_del = Column(Integer, nullable=False, server_default=text("'0'"))
    zamowienia = Column(Integer, nullable=False, server_default=text("'0'"))
    zamowienia_add = Column(Integer, nullable=False, server_default=text("'0'"))
    zamowienia_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    zamowienia_del = Column(Integer, nullable=False, server_default=text("'0'"))
    users = Column(Integer, nullable=False, server_default=text("'0'"))
    users_add = Column(Integer, nullable=False, server_default=text("'0'"))
    users_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    users_del = Column(Integer, nullable=False, server_default=text("'0'"))
    konfiguracja = Column(Integer, nullable=False, server_default=text("'0'"))
    konfiguracja_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    przesylki = Column(Integer, nullable=False, server_default=text("'0'"))
    przesylki_add = Column(Integer, nullable=False, server_default=text("'0'"))
    przesylki_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    przesylki_del = Column(Integer, nullable=False, server_default=text("'0'"))
    news = Column(Integer, nullable=False, server_default=text("'0'"))
    news_add = Column(Integer, nullable=False, server_default=text("'0'"))
    akt = Column(Integer, nullable=False, server_default=text("'0'"))
    akt_add = Column(Integer, nullable=False, server_default=text("'0'"))
    akt_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    akt_del = Column(Integer, nullable=False, server_default=text("'0'"))
    faktury = Column(Integer, nullable=False, server_default=text("'0'"))
    faktury_add = Column(Integer, nullable=False, server_default=text("'0'"))
    faktury_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    faktury_del = Column(Integer, nullable=False, server_default=text("'0'"))
    pliki = Column(Integer, nullable=False, server_default=text("'0'"))
    pliki_add = Column(Integer, nullable=False, server_default=text("'0'"))
    pliki_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    pliki_del = Column(Integer, nullable=False, server_default=text("'0'"))
    rabaty = Column(Integer, nullable=False, server_default=text("'0'"))
    rabaty_add = Column(Integer, nullable=False, server_default=text("'0'"))
    rabaty_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    rabaty_del = Column(Integer, nullable=False, server_default=text("'0'"))
    menu = Column(Integer, nullable=False, server_default=text("'0'"))
    menu_add = Column(Integer, nullable=False, server_default=text("'0'"))
    menu_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    menu_del = Column(Integer, nullable=False, server_default=text("'0'"))
    waluta = Column(Integer, nullable=False, server_default=text("'0'"))
    waluta_add = Column(Integer, nullable=False, server_default=text("'0'"))
    waluta_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    waluta_del = Column(Integer, nullable=False, server_default=text("'0'"))
    komunikaty = Column(Integer, nullable=False, server_default=text("'0'"))
    komunikaty_add = Column(Integer, nullable=False, server_default=text("'0'"))
    komunikaty_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    komunikaty_del = Column(Integer, nullable=False, server_default=text("'0'"))
    dostepnosc = Column(Integer, nullable=False, server_default=text("'0'"))
    dostepnosc_add = Column(Integer, nullable=False, server_default=text("'0'"))
    dostepnosc_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    dostepnosc_del = Column(Integer, nullable=False, server_default=text("'0'"))
    statystyki = Column(Integer, nullable=False, server_default=text("'0'"))
    sonda = Column(Integer, nullable=False, server_default=text("'0'"))
    sonda_add = Column(Integer, nullable=False, server_default=text("'0'"))
    sonda_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    sonda_del = Column(Integer, nullable=False, server_default=text("'0'"))
    recenzje = Column(Integer, nullable=False, server_default=text("'0'"))
    recenzje_del = Column(Integer, nullable=False, server_default=text("'0'"))
    subskrypcja = Column(Integer, nullable=False, server_default=text("'0'"))
    subskrypcja_add = Column(Integer, nullable=False, server_default=text("'0'"))
    subskrypcja_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    subskrypcja_del = Column(Integer, nullable=False, server_default=text("'0'"))
    dzialy = Column(Integer, nullable=False, server_default=text("'0'"))
    dzialy_add = Column(Integer, nullable=False, server_default=text("'0'"))
    dzialy_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    dzialy_del = Column(Integer, nullable=False, server_default=text("'0'"))
    dostawcy = Column(Integer, nullable=False, server_default=text("'0'"))
    dostawcy_add = Column(Integer, nullable=False, server_default=text("'0'"))
    dostawcy_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    dostawcy_del = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_konfiguracja = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_konfiguracja_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_uzytkownicy = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_uzytkownicy_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_uzytkownicy_del = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_zakupy = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_zakupy_del = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_prowizje = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_prowizje_del = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_wyplaty = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_wyplaty_add = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_wyplaty_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_wyplaty_del = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_odwiedziny = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_kody = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_kody_add = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_kody_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    PP_kody_del = Column(Integer, nullable=False, server_default=text("'0'"))
    onet = Column(Integer, nullable=False, server_default=text("'0'"))
    onet_add = Column(Integer, nullable=False, server_default=text("'0'"))
    kody_rabatowe = Column(Integer, nullable=False, server_default=text("'0'"))
    kody_rabatowe_add = Column(Integer, nullable=False, server_default=text("'0'"))
    kody_rabatowe_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    kody_rabatowe_del = Column(Integer, nullable=False, server_default=text("'0'"))
    rachunki = Column(Integer, nullable=False, server_default=text("'0'"))
    rachunki_add = Column(Integer, nullable=False, server_default=text("'0'"))
    rachunki_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    rachunki_del = Column(Integer, nullable=False, server_default=text("'0'"))
    person = Column(Integer, nullable=False, server_default=text("'0'"))
    person_add = Column(Integer, nullable=False, server_default=text("'0'"))
    person_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    person_del = Column(Integer, nullable=False, server_default=text("'0'"))
    recenzje_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    magazyn = Column(Integer, nullable=False, server_default=text("'0'"))
    magazyn_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    pokrewne = Column(Integer, nullable=False, server_default=text("'0'"))
    pokrewne_add = Column(Integer, nullable=False, server_default=text("'0'"))
    pokrewne_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    pokrewne_del = Column(Integer, nullable=False, server_default=text("'0'"))
    logowanie_praca_od = Column(Integer, nullable=False, server_default=text("'0'"))
    logowanie_praca_do = Column(Integer, nullable=False, server_default=text("'0'"))
    poczta = Column(Integer, nullable=False, server_default=text("'0'"))
    poczta_del = Column(Integer, nullable=False, server_default=text("'0'"))
    korekty = Column(Integer, nullable=False, server_default=text("'0'"))
    korekty_add = Column(Integer, nullable=False, server_default=text("'0'"))
    korekty_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    korekty_del = Column(Integer, nullable=False, server_default=text("'0'"))
    paragony = Column(Integer, nullable=False, server_default=text("'0'"))
    paragony_add = Column(Integer, nullable=False, server_default=text("'0'"))
    paragony_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    paragony_del = Column(Integer, nullable=False, server_default=text("'0'"))
    klienci_add = Column(Integer, nullable=False, server_default=text("'0'"))
    statusy_zamowien = Column(Integer, nullable=False, server_default=text("'0'"))
    statusy_zamowien_add = Column(Integer, nullable=False, server_default=text("'0'"))
    statusy_zamowien_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    statusy_zamowien_del = Column(Integer, nullable=False, server_default=text("'0'"))
    menu_01 = Column(Integer, nullable=False, server_default=text("'0'"))
    menu_02 = Column(Integer, nullable=False, server_default=text("'0'"))
    menu_03 = Column(Integer, nullable=False, server_default=text("'0'"))
    menu_04 = Column(Integer, nullable=False, server_default=text("'0'"))
    menu_05 = Column(Integer, nullable=False, server_default=text("'0'"))
    menu_06 = Column(Integer, nullable=False, server_default=text("'0'"))
    menu_07 = Column(Integer, nullable=False, server_default=text("'0'"))
    menu_08 = Column(Integer, nullable=False, server_default=text("'0'"))
    menu_09 = Column(Integer, nullable=False, server_default=text("'0'"))
    korespondencja = Column(Integer, nullable=False, server_default=text("'0'"))
    korespondencja_add = Column(Integer, nullable=False, server_default=text("'0'"))
    korespondencja_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    korespondencja_del = Column(Integer, nullable=False, server_default=text("'0'"))
    menu_10 = Column(Integer, nullable=False, server_default=text("'0'"))
    lojalka = Column(Integer, nullable=False, server_default=text("'0'"))
    lojalka_add = Column(Integer, nullable=False, server_default=text("'0'"))
    lojalka_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    lojalka_del = Column(Integer, nullable=False, server_default=text("'0'"))
    kody_kwotowe = Column(Integer, nullable=False, server_default=text("'0'"))
    kody_kwotowe_add = Column(Integer, nullable=False, server_default=text("'0'"))
    kody_kwotowe_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    kody_kwotowe_del = Column(Integer, nullable=False, server_default=text("'0'"))
    akcesoria = Column(Integer, nullable=False, server_default=text("'0'"))
    akcesoria_add = Column(Integer, nullable=False, server_default=text("'0'"))
    akcesoria_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    akcesoria_del = Column(Integer, nullable=False, server_default=text("'0'"))
    zam_platnosc_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    zam_produkcja_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    zam_komplet_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    zam_wysylka_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    magazyn_pz = Column(Integer, nullable=False, server_default=text("'0'"))
    magazyn_pz_add = Column(Integer, nullable=False, server_default=text("'0'"))
    magazyn_pz_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    magazyn_pz_del = Column(Integer, nullable=False, server_default=text("'0'"))
    magazyn_wz = Column(Integer, nullable=False, server_default=text("'0'"))
    magazyn_wz_add = Column(Integer, nullable=False, server_default=text("'0'"))
    magazyn_wz_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    magazyn_wz_del = Column(Integer, nullable=False, server_default=text("'0'"))
    backorder = Column(Integer, nullable=False, server_default=text("'0'"))
    koszyki_klientow = Column(Integer, nullable=False, server_default=text("'0'"))
    zamowienia_dane_klienta = Column(Integer, nullable=False, server_default=text("'0'"))
    zamowienia_ostatnie_dni = Column(SmallInteger, nullable=False, server_default=text("'0'"))
    zamowienia_historia = Column(Integer, nullable=False, server_default=text("'0'"))
    wywieszki = Column(Integer, nullable=False, server_default=text("'0'"))
    wywieszki_add = Column(Integer, nullable=False, server_default=text("'0'"))
    wywieszki_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    wywieszki_del = Column(Integer, nullable=False, server_default=text("'0'"))
    notatki_crm = Column(Integer, nullable=False, server_default=text("'0'"))
    notatki_crm_add = Column(Integer, nullable=False, server_default=text("'0'"))
    notatki_crm_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    notatki_crm_del = Column(Integer, nullable=False, server_default=text("'0'"))
    aktywnosc = Column(Integer, nullable=False, server_default=text("'0'"))
    obsluga_klienta = Column(Integer, nullable=False, server_default=text("'0'"))
    telefon = Column(String(25), nullable=False, server_default=text("''"))
    branze = Column(Integer, nullable=False, server_default=text("'0'"))
    branze_add = Column(Integer, nullable=False, server_default=text("'0'"))
    branze_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    branze_del = Column(Integer, nullable=False, server_default=text("'0'"))
    klienci_pliki = Column(Integer, nullable=False, server_default=text("'0'"))
    klienci_pliki_add = Column(Integer, nullable=False, server_default=text("'0'"))
    klienci_pliki_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    klienci_pliki_del = Column(Integer, nullable=False, server_default=text("'0'"))
    zaopatrzenie = Column(Integer, nullable=False, server_default=text("'0'"))
    zaopatrzenie_add = Column(Integer, nullable=False, server_default=text("'0'"))
    zaopatrzenie_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    zaopatrzenie_del = Column(Integer, nullable=False, server_default=text("'0'"))
    import_sz = Column(Integer, nullable=False, server_default=text("'0'"))
    import_sz_add = Column(Integer, nullable=False, server_default=text("'0'"))
    import_sz_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    import_sz_del = Column(Integer, nullable=False, server_default=text("'0'"))
    kraje = Column(Integer, nullable=False, server_default=text("'0'"))
    kraje_add = Column(Integer, nullable=False, server_default=text("'0'"))
    kraje_edit = Column(Integer, nullable=False, server_default=text("'0'"))
    kraje_del = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsWaluta(Base):
    __tablename__ = 'kqs_waluta'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(50), nullable=False, server_default=text("''"))
    oznaczenie = Column(String(15), nullable=False, server_default=text("''"))
    kurs = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    kod = Column(String(5), nullable=False, server_default=text("''"))


class KqsWeryfikator(Base):
    __tablename__ = 'kqs_weryfikator'

    numer = Column(BigInteger, primary_key=True)
    kod = Column(String(100), nullable=False, server_default=text("''"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    user_ip = Column(String(20), nullable=False, server_default=text("''"))
    bezpiecznik = Column(Integer, nullable=False, server_default=text("'0'"))
    opornik = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsWywieszki(Base):
    __tablename__ = 'kqs_wywieszki'

    numer = Column(BigInteger, primary_key=True)
    nazwa = Column(String(100), nullable=False, server_default=text("''"))
    wywieszka = Column(Text, nullable=False)


t_kqs_wywieszki_kategorie = Table(
    'kqs_wywieszki_kategorie', metadata,
    Column('wywieszka_id', BigInteger, nullable=False, index=True, server_default=text("'0'")),
    Column('kategoria_id', BigInteger, nullable=False, index=True, server_default=text("'0'"))
)


t_kqs_zakupione_razem = Table(
    'kqs_zakupione_razem', metadata,
    Column('produkt_id1', BigInteger, nullable=False, server_default=text("'0'")),
    Column('produkt_id2', BigInteger, nullable=False, server_default=text("'0'")),
    Index('produkt_id12', 'produkt_id1', 'produkt_id2')
)


class KqsZalaczniki(Base):
    __tablename__ = 'kqs_zalaczniki'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    katalog = Column(String(50), nullable=False, server_default=text("''"))
    plik_rozmiar = Column(BigInteger, nullable=False, server_default=text("'0'"))
    plik_typ = Column(String(50), nullable=False, server_default=text("''"))
    plik_nazwa = Column(String(50), nullable=False, server_default=text("''"))
    plik_nazwa_zastepcza = Column(String(50), nullable=False, server_default=text("''"))


class KqsZamowienia(Base):
    __tablename__ = 'kqs_zamowienia'

    id = Column(BigInteger, primary_key=True)
    cena = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    przesylka = Column(BigInteger, nullable=False, server_default=text("'0'"))
    u_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    status = Column(Integer, nullable=False, index=True, server_default=text("'0'"))
    powod = Column(Text, nullable=False)
    waga = Column(Float(8, True), nullable=False, server_default=text("'0.00'"))
    klient_email = Column(String(100), nullable=False, server_default=text("''"))
    klient_telefon = Column(String(20), nullable=False, server_default=text("''"))
    waluta_jednostka = Column(String(15), nullable=False, server_default=text("''"))
    waluta_przelicznik = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    uwagi = Column(Text, nullable=False)
    user_kup_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    admin_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    last_admin_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    last_admin_data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    status_wplata = Column(String(11), nullable=False, server_default=text("''"))
    status_produkcja = Column(String(11), nullable=False, server_default=text("''"))
    status_wyprodukowano = Column(String(11), nullable=False, server_default=text("''"))
    numer_listu = Column(String(50), nullable=False, server_default=text("''"))
    data_wyslania = Column(BigInteger, nullable=False, server_default=text("'0'"))
    dokument = Column(Integer, nullable=False, server_default=text("'0'"))
    przesylka_nazwa = Column(String(100), nullable=False, server_default=text("''"))
    przesylka_koszt = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    przesylka_koszt_brutto = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    przesylka_opis = Column(Text, nullable=False)
    przesylka_fra = Column(Integer, nullable=False, server_default=text("'0'"))
    kwota_platnosci = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    przelew = Column(Integer, nullable=False, server_default=text("'0'"))
    gotowka = Column(Integer, nullable=False, server_default=text("'0'"))
    potwierdzenie = Column(Integer, nullable=False, server_default=text("'0'"))
    klient_osoba = Column(String(100), nullable=False)
    klient_firma = Column(String(255), nullable=False)
    klient_nip = Column(String(50), nullable=False)
    klient_wojewodztwo = Column(String(100), nullable=False)
    klient_kraj = Column(String(100), nullable=False)
    rabat_model = Column(Integer, nullable=False, server_default=text("'0'"))
    status_dodatkowy = Column(BigInteger, nullable=False, server_default=text("'0'"))
    przesylka_vat = Column(Integer, nullable=False, server_default=text("'0'"))
    klient_ulica = Column(String(100), nullable=False)
    klient_dom = Column(String(20), nullable=False, server_default=text("''"))
    klient_kod = Column(String(10), nullable=False, server_default=text("''"))
    klient_miasto = Column(String(100), nullable=False, server_default=text("''"))
    wysylka_odbiorca = Column(String(200), nullable=False, server_default=text("''"))
    wysylka_ulica = Column(String(100), nullable=False, server_default=text("''"))
    wysylka_dom = Column(String(20), nullable=False, server_default=text("''"))
    wysylka_kod = Column(String(10), nullable=False, server_default=text("''"))
    wysylka_miasto = Column(String(100), nullable=False, server_default=text("''"))
    referent = Column(String(255), nullable=False, server_default=text("''"))
    bon_kod = Column(String(35), nullable=False, server_default=text("''"))
    bon_wartosc = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    status_przypomnienie = Column(String(11), nullable=False, server_default=text("''"))
    unikalny_numer = Column(String(20), nullable=False, server_default=text("''"))
    klient_ip = Column(String(50), nullable=False, server_default=text("''"))
    waluta_kod = Column(String(5), nullable=False, server_default=text("''"))
    odroczenie = Column(Integer, nullable=False, server_default=text("'0'"))
    leasing = Column(Integer, nullable=False, server_default=text("'0'"))
    jezyk = Column(String(32), nullable=False, server_default=text("''"))
    rodzaj_cen = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsZamowieniaBackorder(Base):
    __tablename__ = 'kqs_zamowienia_backorder'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    ilosc = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    dostepnosc = Column(String(255), nullable=False, server_default=text("''"))


class KqsZamowieniaGabaryty(Base):
    __tablename__ = 'kqs_zamowienia_gabaryty'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    waga = Column(Float(8, True), nullable=False, server_default=text("'0.00'"))
    wysokosc = Column(Integer, nullable=False, server_default=text("'0'"))
    szerokosc = Column(Integer, nullable=False, server_default=text("'0'"))
    dlugosc = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsZamowieniaHistoria(Base):
    __tablename__ = 'kqs_zamowienia_historia'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    komunikat = Column(String(255), nullable=False, server_default=text("''"))
    admin_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    admin_nazwa = Column(String(100), nullable=False, server_default=text("''"))
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsZamowieniaHistoriaProdukty(Base):
    __tablename__ = 'kqs_zamowienia_historia_produkty'

    numer = Column(BigInteger, primary_key=True)
    historia_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt_nazwa = Column(String(200), nullable=False, server_default=text("''"))
    ilosc = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    jm = Column(String(50), nullable=False, server_default=text("''"))
    cena = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    podatek = Column(Integer, nullable=False, server_default=text("'0'"))
    atrybuty = Column(Text, nullable=False)
    atrybuty_magazyn = Column(Text, nullable=False)
    pkwiu = Column(String(50), nullable=False, server_default=text("''"))
    kod_produktu = Column(String(50), nullable=False, server_default=text("''"))
    kod_dostawcy = Column(String(50), nullable=False, server_default=text("''"))
    kod_kreskowy = Column(String(25), nullable=False, server_default=text("''"))
    kod_plu = Column(String(25), nullable=False, server_default=text("''"))
    rabat = Column(Float(4, True), nullable=False, server_default=text("'0.00'"))
    kod_rabatowy = Column(String(50), nullable=False, server_default=text("''"))
    kod_producenta = Column(String(50), nullable=False, server_default=text("''"))
    status = Column(Integer, nullable=False, server_default=text("'0'"))


class KqsZamowieniaNotatki(Base):
    __tablename__ = 'kqs_zamowienia_notatki'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    notatka = Column(Text, nullable=False)
    data = Column(BigInteger, nullable=False, server_default=text("'0'"))
    admin_id = Column(BigInteger, nullable=False, server_default=text("'0'"))


class KqsZamowieniaPliki(Base):
    __tablename__ = 'kqs_zamowienia_pliki'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    produkt_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    plik_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    identyfikator = Column(String(100), nullable=False, server_default=text("''"))


class KqsZamowieniaProdukty(Base):
    __tablename__ = 'kqs_zamowienia_produkty'

    numer = Column(BigInteger, primary_key=True)
    zamowienie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt_nazwa = Column(String(200), nullable=False, server_default=text("''"))
    ilosc = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    jm = Column(String(50), nullable=False, server_default=text("''"))
    cena = Column(Float(12, True), nullable=False, server_default=text("'0.00'"))
    podatek = Column(Integer, nullable=False, server_default=text("'0'"))
    atrybuty = Column(Text, nullable=False)
    atrybuty_magazyn = Column(Text, nullable=False)
    pkwiu = Column(String(50), nullable=False, server_default=text("''"))
    kod_produktu = Column(String(50), nullable=False, server_default=text("''"))
    kod_dostawcy = Column(String(50), nullable=False, server_default=text("''"))
    kod_kreskowy = Column(String(50), nullable=False, server_default=text("''"))
    kod_plu = Column(String(50), nullable=False, server_default=text("''"))
    rabat = Column(Float(4, True), nullable=False, server_default=text("'0.00'"))
    kod_rabatowy = Column(String(50), nullable=False, server_default=text("''"))
    kod_producenta = Column(String(50), nullable=False, server_default=text("''"))


class KqsZaopatrzenie(Base):
    __tablename__ = 'kqs_zaopatrzenie'

    numer = Column(BigInteger, primary_key=True)
    nr_dokumentu = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data_wystawienia = Column(BigInteger, nullable=False, server_default=text("'0'"))
    data_dostawy = Column(Date, nullable=False)
    dostawca_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    dostawca_nazwa = Column(String(255), nullable=False, server_default=text("''"))
    adnotacje = Column(Text, nullable=False)
    admin_id = Column(BigInteger, nullable=False, server_default=text("'0'"))
    admin_nazwa = Column(String(100), nullable=False, server_default=text("''"))


class KqsZaopatrzenieProdukty(Base):
    __tablename__ = 'kqs_zaopatrzenie_produkty'

    numer = Column(BigInteger, primary_key=True)
    zaopatrzenie_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt_id = Column(BigInteger, nullable=False, index=True, server_default=text("'0'"))
    produkt_nazwa = Column(String(200), nullable=False, server_default=text("''"))
    ilosc = Column(Float(asdecimal=True), nullable=False, server_default=text("'0'"))
    jm = Column(String(50), nullable=False, server_default=text("''"))
    atrybuty = Column(Text, nullable=False)
    atrybuty_magazyn = Column(Text, nullable=False)
    pkwiu = Column(String(50), nullable=False, server_default=text("''"))
    kod_produktu = Column(String(50), nullable=False, server_default=text("''"))
    kod_dostawcy = Column(String(50), nullable=False, server_default=text("''"))
    kod_producenta = Column(String(50), nullable=False, server_default=text("''"))
    kod_kreskowy = Column(String(25), nullable=False, server_default=text("''"))
    kod_plu = Column(String(25), nullable=False, server_default=text("''"))
