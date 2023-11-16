"""
Demo script using (Py-) MuPDF "Story" feature.

The following features are implemented:

* Use of Story "template" feature to provide row content
* Use database access (SQLITE) to fetch row content
* Use ElementPosition feature to locate cell positions on page
* Simulate feature "Table Header Repeat"
* Simulate feature "Cell Grid Lines"

"""
import io
import sqlite3
import sys
import fitz
'\nTable data. Used to populate a temporary SQL database, which will be processed by the script.\nIts only purpose is to avoid carrying around a separate database file.\n'
table_data = "China;Beijing;21542000;1.5%;2018\nJapan;Tokyo;13921000;11.2%;2019\nDR Congo;Kinshasa;12691000;13.2%;2017\nRussia;Moscow;12655050;8.7%;2021\nIndonesia;Jakarta;10562088;3.9%;2020\nEgypt;Cairo;10107125;9.3%;2022\nSouth Korea;Seoul;9508451;18.3%;2022\nMexico;Mexico City;9209944;7.3%;2020\nUnited Kingdom;London;9002488;13.4%;2020\nBangladesh;Dhaka;8906039;5.3%;2011\nPeru;Lima;8852000;26.3%;2012\nIran;Tehran;8693706;9.9%;2016\nThailand;Bangkok;8305218;11.6%;2010\nVietnam;Hanoi;8053663;8.3%;2019\nIraq;Baghdad;7682136;17.6%;2021\nSaudi Arabia;Riyadh;7676654;21.4%;2018\nHong Kong;Hong Kong;7291600;100%;2022\nColombia;Bogotá;7181469;13.9%;2011\nChile;Santiago;6310000;32.4%;2012\nTurkey;Ankara;5747325;6.8%;2021\nSingapore;Singapore;5453600;91.8%;2021\nAfghanistan;Kabul;4601789;11.5%;2021\nKenya;Nairobi;4397073;8.3%;2019\nJordan;Amman;4061150;36.4%;2021\nAlgeria;Algiers;3915811;8.9%;2011\nGermany;Berlin;3677472;4.4%;2021\nSpain;Madrid;3305408;7.0%;2021\nEthiopia;Addis Ababa;3040740;2.5%;2012\nKuwait;Kuwait City;2989000;70.3%;2018\nGuatemala;Guatemala City;2934841;16.7%;2020\nSouth Africa;Pretoria;2921488;4.9%;2011\nUkraine;Kyiv;2920873;6.7%;2021\nArgentina;Buenos Aires;2891082;6.4%;2010\nNorth Korea;Pyongyang;2870000;11.1%;2016\nUzbekistan;Tashkent;2860600;8.4%;2022\nItaly;Rome;2761632;4.7%;2022\nEcuador;Quito;2800388;15.7%;2020\nCameroon;Yaoundé;2765568;10.2%;2015\nZambia;Lusaka;2731696;14.0%;2020\nSudan;Khartoum;2682431;5.9%;2012\nBrazil;Brasília;2648532;1.2%;2012\nTaiwan;Taipei (de facto);2608332;10.9%;2020\nYemen;Sanaa;2575347;7.8%;2012\nAngola;Luanda;2571861;7.5%;2020\nBurkina Faso;Ouagadougou;2453496;11.1%;2019\nGhana;Accra;2388000;7.3%;2017\nSomalia;Mogadishu;2388000;14.0%;2021\nAzerbaijan;Baku;2303100;22.3%;2022\nCambodia;Phnom Penh;2281951;13.8%;2019\nVenezuela;Caracas;2245744;8.0%;2016\nFrance;Paris;2139907;3.3%;2022\nCuba;Havana;2132183;18.9%;2020\nZimbabwe;Harare;2123132;13.3%;2012\nSyria;Damascus;2079000;9.7%;2019\nBelarus;Minsk;1996553;20.8%;2022\nAustria;Vienna;1962779;22.0%;2022\nPoland;Warsaw;1863056;4.9%;2021\nPhilippines;Manila;1846513;1.6%;2020\nMali;Bamako;1809106;8.3%;2009\nMalaysia;Kuala Lumpur;1782500;5.3%;2019\nRomania;Bucharest;1716983;8.9%;2021\nHungary;Budapest;1706851;17.6%;2022\nCongo;Brazzaville;1696392;29.1%;2015\nSerbia;Belgrade;1688667;23.1%;2021\nUganda;Kampala;1680600;3.7%;2019\nGuinea;Conakry;1660973;12.3%;2014\nMongolia;Ulaanbaatar;1466125;43.8%;2020\nHonduras;Tegucigalpa;1444085;14.0%;2021\nSenegal;Dakar;1438725;8.5%;2021\nNiger;Niamey;1334984;5.3%;2020\nUruguay;Montevideo;1319108;38.5%;2011\nBulgaria;Sofia;1307439;19.0%;2021\nOman;Muscat;1294101;28.6%;2021\nCzech Republic;Prague;1275406;12.1%;2022\nMadagascar;Antananarivo;1275207;4.4%;2018\nKazakhstan;Astana;1239900;6.5%;2022\nNigeria;Abuja;1235880;0.6%;2011\nGeorgia;Tbilisi;1201769;32.0%;2022\nMauritania;Nouakchott;1195600;25.9%;2019\nQatar;Doha;1186023;44.1%;2020\nLibya;Tripoli;1170000;17.4%;2019\nMyanmar;Naypyidaw;1160242;2.2%;2014\nRwanda;Kigali;1132686;8.4%;2012\nMozambique;Maputo;1124988;3.5%;2020\nDominican Republic;Santo Domingo;1111838;10.0%;2010\nArmenia;Yerevan;1096100;39.3%;2021\nKyrgyzstan;Bishkek;1074075;16.5%;2021\nSierra Leone;Freetown;1055964;12.5%;2015\nNicaragua;Managua;1055247;15.4%;2020\nCanada;Ottawa;1017449;2.7%;2021\nPakistan;Islamabad;1014825;0.4%;2017\nLiberia;Monrovia;1010970;19.5%;2008\nUnited Arab Emirates;Abu Dhabi;1010092;10.8%;2020\nMalawi;Lilongwe;989318;5.0%;2018\nHaiti;Port-au-Prince;987310;8.6%;2015\nSweden;Stockholm;978770;9.4%;2021\nEritrea;Asmara;963000;26.6%;2020\nIsrael;Jerusalem;936425;10.5%;2019\nLaos;Vientiane;927724;12.5%;2019\nChad;N'Djamena;916000;5.3%;2009\nNetherlands;Amsterdam;905234;5.2%;2022\nCentral African Republic;Bangui;889231;16.3%;2020\nPanama;Panama City;880691;20.2%;2013\nTajikistan;Dushanbe;863400;8.9%;2020\nNepal;Kathmandu;845767;2.8%;2021\nTogo;Lomé;837437;9.7%;2010\nTurkmenistan;Ashgabat;791000;12.5%;2017\nMoldova;Chişinău;779300;25.5%;2019\nCroatia;Zagreb;769944;19.0%;2021\nGabon;Libreville;703904;30.1%;2013\nNorway;Oslo;697010;12.9%;2021\nMacau;Macau;671900;97.9%;2022\nUnited States;Washington D.C.;670050;0.2%;2021\nJamaica;Kingston;662491;23.4%;2019\nFinland;Helsinki;658864;11.9%;2021\nTunisia;Tunis;638845;5.2%;2014\nDenmark;Copenhagen;638117;10.9%;2021\nGreece;Athens;637798;6.1%;2021\nLatvia;Riga;605802;32.3%;2021\nDjibouti;Djibouti (city);604013;54.6%;2012\nIreland;Dublin;588233;11.8%;2022\nMorocco;Rabat;577827;1.6%;2014\nLithuania;Vilnius;576195;20.7%;2022\nEl Salvador;San Salvador;570459;9.0%;2019\nAlbania;Tirana;557422;19.5%;2011\nNorth Macedonia;Skopje;544086;25.9%;2015\nSouth Sudan;Juba;525953;4.9%;2017\nParaguay;Asunción;521559;7.8%;2020\nPortugal;Lisbon;509614;5.0%;2020\nGuinea-Bissau;Bissau;492004;23.9%;2015\nSlovakia;Bratislava;440948;8.1%;2020\nEstonia;Tallinn;438341;33.0%;2021\nAustralia;Canberra;431380;1.7%;2020\nNamibia;Windhoek;431000;17.0%;2020\nTanzania;Dodoma;410956;0.6%;2012\nPapua New Guinea;Port Moresby;364145;3.7%;2011\nIvory Coast;Yamoussoukro;361893;1.3%;2020\nLebanon;Beirut;361366;6.5%;2014\nBolivia;Sucre;360544;3.0%;2022\nPuerto Rico (US);San Juan;342259;10.5%;2020\nCosta Rica;San José;342188;6.6%;2018\nLesotho;Maseru;330760;14.5%;2016\nCyprus;Nicosia;326739;26.3%;2016\nEquatorial Guinea;Malabo;297000;18.2%;2018\nSlovenia;Ljubljana;285604;13.5%;2021\nEast Timor;Dili;277279;21.0%;2015\nBosnia and Herzegovina;Sarajevo;275524;8.4%;2013\nBahamas;Nassau;274400;67.3%;2016\nBotswana;Gaborone;273602;10.6%;2020\nBenin;Porto-Novo;264320;2.0%;2013\nSuriname;Paramaribo;240924;39.3%;2012\nIndia;New Delhi;249998;0.0%;2011\nSahrawi Arab Democratic Republic;Laayoune (claimed) - Tifariti (de facto);217732 - 3000;—;2014\nNew Zealand;Wellington;217000;4.2%;2021\nBahrain;Manama;200000;13.7%;2020\nKosovo;Pristina;198897;12.0%;2011\nMontenegro;Podgorica;190488;30.3%;2020\nBelgium;Brussels;187686;1.6%;2022\nCape Verde;Praia;159050;27.1%;2017\nMauritius;Port Louis;147066;11.3%;2018\nCuraçao (Netherlands);Willemstad;136660;71.8%;2011\nBurundi;Gitega;135467;1.1%;2020\nSwitzerland;Bern (de facto);134591;1.5%;2020\nTransnistria;Tiraspol;133807;38.5%;2015\nMaldives;Malé;133412;25.6%;2014\nIceland;Reykjavík;133262;36.0%;2021\nLuxembourg;Luxembourg City;124509;19.5%;2021\nGuyana;Georgetown;118363;14.7%;2012\nBhutan;Thimphu;114551;14.7%;2017\nComoros;Moroni;111326;13.5%;2016\nBarbados;Bridgetown;110000;39.1%;2014\nSri Lanka;Sri Jayawardenepura Kotte;107925;0.5%;2012\nBrunei;Bandar Seri Begawan;100700;22.6%;2007\nEswatini;Mbabane;94874;8.0%;2010\nNew Caledonia (France);Nouméa;94285;32.8%;2019\nFiji;Suva;93970;10.2%;2017\nSolomon Islands;Honiara;92344;13.0%;2021\nRepublic of Artsakh;Stepanakert;75000;62.5%;2021\nGambia;Banjul;73000;2.8%;2013\nSão Tomé and Príncipe;São Tomé;71868;32.2%;2015\nKiribati;Tarawa;70480;54.7%;2020\nVanuatu;Port Vila;51437;16.1%;2016\nNorthern Mariana Islands (USA);Saipan;47565;96.1%;2017\nSamoa;Apia;41611;19.0%;2021\nPalestine;Ramallah (de facto);38998;0.8%;2017\nMonaco;Monaco;38350;104.5%;2020\nJersey (UK);Saint Helier;37540;34.2%;2018\nTrinidad and Tobago;Port of Spain;37074;2.4%;2011\nCayman Islands (UK);George Town;34399;50.5%;2021\nGibraltar (UK);Gibraltar;34003;104.1%;2020\nGrenada;St. George's;33734;27.1%;2012\nAruba (Netherlands);Oranjestad;28294;26.6%;2010\nIsle of Man (UK);Douglas;27938;33.2%;2011\nMarshall Islands;Majuro;27797;66.1%;2011\nTonga;Nukuʻalofa;27600;26.0%;2022\nSeychelles;Victoria;26450;24.8%;2010\nFrench Polynesia (France);Papeete;26926;8.9%;2017\nAndorra;Andorra la Vella;22873;28.9%;2022\nFaroe Islands (Denmark);Tórshavn;22738;43.0%;2022\nAntigua and Barbuda;St. John's;22219;23.8%;2011\nBelize;Belmopan;20621;5.2%;2016\nSaint Lucia;Castries;20000;11.1%;2013\nGuernsey (UK);Saint Peter Port;18958;30.1%;2019\nGreenland (Denmark);Nuuk;18800;33.4%;2021\nDominica;Roseau;14725;20.3%;2011\nSaint Kitts and Nevis;Basseterre;14000;29.4%;2018\nSaint Vincent and the Grenadines;Kingstown;12909;12.4%;2012\nBritish Virgin Islands (UK);Road Town;12603;40.5%;2012\nÅland (Finland);Mariehamn;11736;39.0%;2021\nU.S. Virgin Islands (US);Charlotte Amalie;14477;14.5%;2020\nMicronesia;Palikir;6647;5.9%;2010\nTuvalu;Funafuti;6320;56.4%;2017\nMalta;Valletta;5827;1.1%;2019\nLiechtenstein;Vaduz;5774;14.8%;2021\nSaint Pierre and Miquelon (France);Saint-Pierre;5394;91.7%;2019\nCook Islands (NZ);Avarua;4906;28.9%;2016\nSan Marino;City of San Marino;4061;12.0%;2021\nTurks and Caicos Islands (UK);Cockburn Town;3720;8.2%;2016\nAmerican Samoa (USA);Pago Pago;3656;8.1%;2010\nSaint Martin (France);Marigot;3229;10.1%;2017\nSaint Barthélemy (France);Gustavia;2615;24.1%;2010\nFalkland Islands (UK);Stanley;2460;65.4%;2016\nSvalbard (Norway);Longyearbyen;2417;82.2%;2020\nSint Maarten (Netherlands);Philipsburg;1894;4.3%;2011\nChristmas Island (Australia);Flying Fish Cove;1599;86.8%;2016\nAnguilla (UK);The Valley;1067;6.8%;2011\nGuam (US);Hagåtña;1051;0.6%;2010\nWallis and Futuna (France);Mata Utu;1029;8.9%;2018\nBermuda (UK);Hamilton;854;1.3%;2016\nNauru;Yaren (de facto);747;6.0%;2011\nSaint Helena (UK);Jamestown;629;11.6%;2016\nNiue (NZ);Alofi;597;30.8%;2017\nTokelau (NZ);Atafu;541;29.3%;2016\nVatican City;Vatican City (city-state);453;100%;2019\nMontserrat (UK);Brades (de facto) - Plymouth (de jure);449 - 0;-;2011\nNorfolk Island (Australia);Kingston;341;-;2015\nPalau;Ngerulmud;271;1.5%;2010\nCocos (Keeling) Islands (Australia);West Island;134;24.6%;2011\nPitcairn Islands (UK);Adamstown;40;100.0%;2021\nSouth Georgia and the South Sandwich Islands (UK);King Edward Point;22;73.3%;2018"
HTML = '\n    <h1 style="text-align:center">World Capital Cities</h1>\n    <p><i>Percent "%" is city population as a percentage of the country, as of "Year".</i>\n    </p><p></p>\n    <table>\n    <tr id="row">\n        <td id="country"></td>\n        <td id="capital"></td>\n        <td id="population"></td>\n        <td id="percent"></td>\n        <td id="year"></td>\n    </tr>\n    </table>\n'
CSS = '\nbody {\n    font-family: sans-serif;\n}\ntd[id="population"], td[id="percent"], td[id="year"] {\n    text-align: right;\n    padding-right: 2px;\n}'
coords = {}

def recorder(elpos):
    if False:
        for i in range(10):
            print('nop')
    'We only record positions of table rows and cells.\n\n    Information is stored in "coords" with page number as key.\n    '
    global coords
    if elpos.open_close != 2:
        return
    if elpos.id not in ('row', 'country', 'capital', 'population', 'percent', 'year'):
        return
    rect = fitz.Rect(elpos.rect)
    if rect.y1 > elpos.filled:
        return
    (x, y, x1, y0) = coords.get(elpos.page, (set(), set(), 0, sys.maxsize))
    if elpos.id != 'row':
        x.add(rect.x0)
        if rect.x1 > x1:
            x1 = rect.x1
    else:
        y.add(rect.y1)
        if rect.y0 < y0:
            y0 = rect.y0
    coords[elpos.page] = (x, y, x1, y0)
    return
dbfilename = ':memory:'
database = sqlite3.connect(dbfilename)
cursor = database.cursor()
cursor.execute('CREATE TABLE capitals (Country text, Capital text, Population text, Percent text, Year text)')
for value in table_data.splitlines():
    cursor.execute('INSERT INTO capitals VALUES (?,?,?,?,?)', value.split(';'))
select = 'SELECT * FROM capitals ORDER BY "Country" '
story = fitz.Story(HTML, user_css=CSS)
body = story.body
template = body.find(None, 'id', 'row')
table = body.find('table', None, None)
cursor.execute(select)
rows = cursor.fetchall()
database.close()
for (country, capital, population, percent, year) in rows:
    row = template.clone()
    row.find(None, 'id', 'country').add_text(country)
    row.find(None, 'id', 'capital').add_text(capital)
    row.find(None, 'id', 'population').add_text(population)
    row.find(None, 'id', 'percent').add_text(percent)
    row.find(None, 'id', 'year').add_text(year)
    table.append_child(row)
template.remove()
fp = io.BytesIO()
writer = fitz.DocumentWriter(fp)
mediabox = fitz.paper_rect('letter')
where = mediabox + (36, 36, -36, -72)
more = True
page = 0
while more:
    dev = writer.begin_page(mediabox)
    if page > 0:
        delta = (0, 20, 0, 0)
    else:
        delta = (0, 0, 0, 0)
    (more, filled) = story.place(where + delta)
    story.element_positions(recorder, {'page': page, 'filled': where.y1})
    story.draw(dev)
    writer.end_page()
    page += 1
writer.close()
doc = fitz.open('pdf', fp)
for page in doc:
    page.wrap_contents()
    (x, y, x1, y0) = coords[page.number]
    x = sorted(list(x)) + [x1]
    y = [y0] + sorted(list(y))
    shape = page.new_shape()
    for item in y:
        shape.draw_line((x[0] - 2, item), (x[-1] + 2, item))
    for i in range(len(y)):
        if i % 2:
            rect = (x[0] - 2, y[i - 1], x[-1] + 2, y[i])
            shape.draw_rect(rect)
    for i in range(len(x)):
        d = 2 if i == len(x) - 1 else -2
        shape.draw_line((x[i] + d, y[0]), (x[i] + d, y[-1]))
    y0 -= 5
    shape.insert_text((x[0], y0), 'Country', fontname='hebo', fontsize=12)
    shape.insert_text((x[1], y0), 'Capital', fontname='hebo', fontsize=12)
    shape.insert_text((x[2], y0), 'Population', fontname='hebo', fontsize=12)
    shape.insert_text((x[3], y0), '  %', fontname='hebo', fontsize=12)
    shape.insert_text((x[4], y0), 'Year', fontname='hebo', fontsize=12)
    y0 = page.rect.height - 50
    bbox = fitz.Rect(0, y0, page.rect.width, y0 + 20)
    page.insert_textbox(bbox, f'World Capital Cities, Page {page.number + 1} of {doc.page_count}', align=fitz.TEXT_ALIGN_CENTER)
    shape.finish(width=0.3, color=0.5, fill=0.9)
    shape.commit(overlay=False)
doc.subset_fonts()
doc.save(__file__.replace('.py', '.pdf'), deflate=True, garbage=4, pretty=True)
doc.close()