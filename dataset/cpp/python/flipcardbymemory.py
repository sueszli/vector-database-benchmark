'''
Function:
    记忆翻牌小游戏
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import pygame
import random
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk


'''配置类'''
class Config():
    # 根目录
    rootdir = os.path.split(os.path.abspath(__file__))[0]
    # 图片路径
    IMAGEPATHS = {
        'cards_back': 'R0lGODdhdgB6APcAAAAAAAMBGiABAhMDAgsEBAUGCgoGCgYJCAoJBiwJAjQKAQsLCxMLBQsNEhINDBIOEhsOBQ4RExERDRsSCxQTFCUTBSQUCywUBEMUBRMVGSsVC'
                    'joVBBoWFDMWBhkXGRYZFhsZFTUZBhYaHCQaEywaDDQaCjsaBhsbGysbBSscE0McBkscBVQcBC0dHDwdCkMdCh0eIiEeHUkeCjQfE1AfCSIhHT0hDR0iIyoiGDQiCUQ'
                    'iC0UiBkwiBiMjIzsjBTwjE0sjC1QjBVojBTYlGkQlEkslEVMlCyomI1smCmMmCWMoBTcpIlQpEioqKkoqDFUqBVUqDFsqBlwqDDsrG0wrFWMrDEQsGEYsC1wsEmMtE'
                    'motDFEuGGouEWsuBHMuEXMwA2QxGjcyL1syFmQyDWQyE2wyDEwzH1MzG2szE3MzDXs0Cls1C2s1GnM1FHs1E0o4B2Q6AW06FHE6G3M6FHM7CYM7ClI8IF08Hm48IXs8'
                    'FHw8DIM8FHs9Gnw9A0BAQFJALXVBFIpBC4tBFHxCG3RDG3xDFIRDFZJDFGJEJXFEJoNEGoVECYxEG5NEDH5FIo9FI29JE2BKKIJKCYJKI29LK4RLGYxLGZNLI5tLGE9'
                    'MS05NU4xNI4JOLI1OC1hPRJNPE5RPG31RNoxRK6VRG4xSGppSI5NTJZtTGpxTLIdUNW1WNqRWJnFXEmlZKHRaKp5aI4BbJ4FbOaBbLY9cOaxcIJRdJ5FeGG9hR6BjL61'
                    'jK6JkJLFkJGVlZqBmOn5nNX9nJpBnRHNoPa5oM4NpP4VqU45qOL5qL55rR6RsMo9tJ61tO5xxI390NbJ0OpB2UZp2QLt2O8V2OI53Ma93RIp4QLp4RJ95U3t7e5B+ZcW'
                    'ARJODRaKDUJGETcWES7GHTZOINq6JY6KLOMuNU6CORKuPPpWQhJ+RTqSWYsmWZLGXUb2XY7GbX7Sbfq2jVLmkQaunYbOoYb+oZM6rg7iugLmub8iubrKxsrixlsm+g8jIy'
                    'ODf3uXl5Orq6/f29v///wAAACH5BAkKAP8AIf8LSUNDUkdCRzEwMTL/AAAMSExpbm8CEAAAbW50clJHQiBYWVogB84AAgAJAAYAMQAAYWNzcE1TRlQAAAAASUVDIHNSR0IAA'
                    'AAAAAAAAAAAAAAAAPbWAAEAAAAA0y1IUCAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARY3BydAAAAVAAAAAzZGVzYwAAAYQAAABsd3RwdAA'
                    'AAfAAAAAUYmtwdAAAAgQAAAAUclhZWgAAAhgAAAAUZ1hZWgAAAiwAAAAUYlhZWgAAAkAAAAAUZG1uZAAAAlQAAABwZG1kZAAAAsQAAACIdnVlZAAAA0wAAACGdmll/3cAAAPUAAAAJG'
                    'x1bWkAAAP4AAAAFG1lYXMAAAQMAAAAJHRlY2gAAAQwAAAADHJUUkMAAAQ8AAAIDGdUUkMAAAQ8AAAIDGJUUkMAAAQ8AAAIDHRleHQAAAAAQ29weXJpZ2h0IChjKSAxOTk4IEhld2xldHQ'
                    'tUGFja2FyZCBDb21wYW55AABkZXNjAAAAAAAAABJzUkdCIElFQzYxOTY2LTIuMQAAAAAAAAAAAAAAEnNSR0IgSUVDNjE5NjYtMi4xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
                    'AAAAAAAAAAAAAAAAAAAAAAAAABYWVogAAAAAAAA81EAAf8AAAABFsxYWVogAAAAAAAAAAAAAAAAAAAAAFhZWiAAAAAAAABvogAAOPUAAAOQWFlaIAAAAAAAAGKZAAC3hQAAGNpYWVogAAAAA'
                    'AAAJKAAAA+EAAC2z2Rlc2MAAAAAAAAAFklFQyBodHRwOi8vd3d3LmllYy5jaAAAAAAAAAAAAAAAFklFQyBodHRwOi8vd3d3LmllYy5jaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
                    'AAAAAAAAAAAAAAAAAAAABkZXNjAAAAAAAAAC5JRUMgNjE5NjYtMi4xIERlZmF1bHQgUkdCIGNvbG91ciBzcGFjZSAtIHNSR0L/AAAAAAAAAAAAAAAuSUVDIDYxOTY2LTIuMSBEZWZhdWx0IFJHQiBjb2'
                    'xvdXIgc3BhY2UgLSBzUkdCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGRlc2MAAAAAAAAALFJlZmVyZW5jZSBWaWV3aW5nIENvbmRpdGlvbiBpbiBJRUM2MTk2Ni0yLjEAAAAAAAAAAAAAACxSZWZlcmVuY2UgVm'
                    'lld2luZyBDb25kaXRpb24gaW4gSUVDNjE5NjYtMi4xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB2aWV3AAAAAAATpP4AFF8uABDPFAAD7cwABBMLAANcngAAAAFYWVog/wAAAAAATAlWAFAAAABXH+dtZWFzAAAAAAAAAA'
                    'EAAAAAAAAAAAAAAAAAAAAAAAACjwAAAAJzaWcgAAAAAENSVCBjdXJ2AAAAAAAABAAAAAAFAAoADwAUABkAHgAjACgALQAyADcAOwBAAEUASgBPAFQAWQBeAGMAaABtAHIAdwB8AIEAhgCLAJAAlQCaAJ8ApACpAK4AsgC3ALw'
                    'AwQDGAMsA0ADVANsA4ADlAOsA8AD2APsBAQEHAQ0BEwEZAR8BJQErATIBOAE+AUUBTAFSAVkBYAFnAW4BdQF8AYMBiwGSAZoBoQGpAbEBuQHBAckB0QHZAeEB6QHyAfoCAwIMAv8UAh0CJgIvAjgCQQJLAlQCXQJnAnECegKEAo'
                    '4CmAKiAqwCtgLBAssC1QLgAusC9QMAAwsDFgMhAy0DOANDA08DWgNmA3IDfgOKA5YDogOuA7oDxwPTA+AD7AP5BAYEEwQgBC0EOwRIBFUEYwRxBH4EjASaBKgEtgTEBNME4QTwBP4FDQUcBSsFOgVJBVgFZwV3BYYFlgWmBbUFxQXVB'
                    'eUF9gYGBhYGJwY3BkgGWQZqBnsGjAadBq8GwAbRBuMG9QcHBxkHKwc9B08HYQd0B4YHmQesB78H0gflB/gICwgfCDIIRghaCG4IggiWCKoIvgjSCOcI+wkQCSUJOglPCWT/CXkJjwmkCboJzwnlCfsKEQonCj0KVApqCoEKmAquCsUK3'
                    'ArzCwsLIgs5C1ELaQuAC5gLsAvIC+EL+QwSDCoMQwxcDHUMjgynDMAM2QzzDQ0NJg1ADVoNdA2ODakNww3eDfgOEw4uDkkOZA5/DpsOtg7SDu4PCQ8lD0EPXg96D5YPsw/PD+wQCRAmEEMQYRB+EJsQuRDXEPURExExEU8RbRGMEaoRyR'
                    'HoEgcSJhJFEmQShBKjEsMS4xMDEyMTQxNjE4MTpBPFE+UUBhQnFEkUahSLFK0UzhTwFRIVNBVWFXgVmxW9FeAWAxYmFkkWbBaPFrIW1hb6Fx0XQRdlF4kX/64X0hf3GBsYQBhlGIoYrxjVGPoZIBlFGWsZkRm3Gd0aBBoqGlEadxqeGsU'
                    'a7BsUGzsbYxuKG7Ib2hwCHCocUhx7HKMczBz1HR4dRx1wHZkdwx3sHhYeQB5qHpQevh7pHxMfPh9pH5Qfvx/qIBUgQSBsIJggxCDwIRwhSCF1IaEhziH7IiciVSKCIq8i3SMKIzgjZiOUI8Ij8CQfJE0kfCSrJNolCSU4JWgllyXHJfcmJy'
                    'ZXJocmtyboJxgnSSd6J6sn3CgNKD8ocSiiKNQpBik4KWspnSnQKgIqNSpoKpsqzysCKzYraSudK9EsBSw5LG4soizXLQwtQS12Last4f8uFi5MLoIuty7uLyQvWi+RL8cv/jA1MGwwpDDbMRIxSjGCMbox8jIqMmMymzLUMw0zRjN/M7gz8T'
                    'QrNGU0njTYNRM1TTWHNcI1/TY3NnI2rjbpNyQ3YDecN9c4FDhQOIw4yDkFOUI5fzm8Ofk6Njp0OrI67zstO2s7qjvoPCc8ZTykPOM9Ij1hPaE94D4gPmA+oD7gPyE/YT+iP+JAI0BkQKZA50EpQWpBrEHuQjBCckK1QvdDOkN9Q8BEA0RHRI'
                    'pEzkUSRVVFmkXeRiJGZ0arRvBHNUd7R8BIBUhLSJFI10kdSWNJqUnwSjdKfUrESwxLU0uaS+JMKkxyTLpNAk3/Sk2TTdxOJU5uTrdPAE9JT5NP3VAnUHFQu1EGUVBRm1HmUjFSfFLHUxNTX1OqU/ZUQlSPVNtVKFV1VcJWD1ZcVqlW91dEV5J'
                    'X4FgvWH1Yy1kaWWlZuFoHWlZaplr1W0VblVvlXDVchlzWXSddeF3JXhpebF69Xw9fYV+zYAVgV2CqYPxhT2GiYfViSWKcYvBjQ2OXY+tkQGSUZOllPWWSZedmPWaSZuhnPWeTZ+loP2iWaOxpQ2maafFqSGqfavdrT2una/9sV2yvbQhtYG25b'
                    'hJua27Ebx5veG/RcCtwhnDgcTpxlXHwcktypnMBc11zuHQUdHB0zHUodYV14XY+/3abdvh3VnezeBF4bnjMeSp5iXnnekZ6pXsEe2N7wnwhfIF84X1BfaF+AX5ifsJ/I3+Ef+WAR4CogQqBa4HNgjCCkoL0g1eDuoQdhICE44VHhauGDoZyhte'
                    'HO4efiASIaYjOiTOJmYn+imSKyoswi5aL/IxjjMqNMY2Yjf+OZo7OjzaPnpAGkG6Q1pE/kaiSEZJ6kuOTTZO2lCCUipT0lV+VyZY0lp+XCpd1l+CYTJi4mSSZkJn8mmia1ZtCm6+cHJyJnPedZJ3SnkCerp8dn4uf+qBpoNihR6G2oiailqMGo'
                    '3aj5qRWpMelOKWpphqmi6b9p26n4KhSqMSpN6mpqv8cqo+rAqt1q+msXKzQrUStuK4trqGvFq+LsACwdbDqsWCx1rJLssKzOLOutCW0nLUTtYq2AbZ5tvC3aLfguFm40blKucK6O7q1uy67p7whvJu9Fb2Pvgq+hL7/v3q/9cBwwOzBZ8Hjwl/C'
                    '28NYw9TEUcTOxUvFyMZGxsPHQce/yD3IvMk6ybnKOMq3yzbLtsw1zLXNNc21zjbOts83z7jQOdC60TzRvtI/0sHTRNPG1EnUy9VO1dHWVdbY11zX4Nhk2OjZbNnx2nba+9uA3AXcit0Q3ZbeHN6i3ynfr+A24L3hROHM4lPi2+Nj4+vkc+T85YTm'
                    'DeaW5x/nqegy6LxU6Ubp0Opb6uXrcOv77IbtEe2c7ijutO9A78zwWPDl8XLx//KM8xnzp/Q09ML1UPXe9m32+/eK+Bn4qPk4+cf6V/rn+3f8B/yY/Sn9uv5L/tz/bf//ACwAAAAAdgB6AAAI/wD9CRxIsKDBgwgTKlzIsKFDf/0ERnx4cCLFixgz'
                    'ahxocaPHjyBDPuS3j6TJfShLoiSpsmXKlCz5sYS5UubKlzZdxlR5kubNniWBxjTZcaE+e9g0+VmaSZPTplAzNXVKVZNUT0+rWpUqVavXr2C9Xs0a9ulUqlL98Cpnr6G9PwDiyp1Lt67du3jz6t3Ld++1fkUJ5gMQoAcICogTZ6BwQsSHxBQyLM4gQ'
                    'sRixJIha46cWbKHyIoxM+ZwonRpxBxQI45xwkMGD583y57NGEYPAH8hHiwHoAbd0w0oi7ghgsKHCMgXFGgQ/MTkyxQKxC0QwfICA4gRAFiwAMCBAwjAL/94IEECg7ggJnAgAILBAAIAIECgO8DBhAkOGCDovqDBaAkHlHfAY/zF1cBt+iTEywIcyL'
                    'UAYyc0IAIMB0Rwww093FDDYRIsQIABD3gQAwjMkcZBAQs4wMEDiDmAgAMUgKDddvAdwF0BBCzAwAVD/GEBAepxMIF8JFhhByKuUEONHT/gMIFcEpDWxJRN1MDBg1d+0EMP3UlnAAwU5JPQNQB40B0AjJn5IAwZnNBDBCNs6AABMOY3pwMSvAeABCMI'
                    'EBcBgBIgJAQDMGDBCD8ywIADiN43ghlm3HILBAxUMMIQM0xAQiSR+LKON8/8Ugw1zLiiCpMzcBDDA69REEMTEpz/hlgPJxhgQAEFwFCAmAiR6YFcjKF2AggcNBHGEhVMkQMKM1gAgQXQphDtDDOUMMUfVpBQwbYVkDDDAANo8AOmJGhwgQYpkDACCT9Q'
                    '0oozdljQ7BJhxOWCK9akYw0rkEACSzHSjOrLwK640kokdkzRQxhNgHACBRI0UBquuN7G60FknhBXigt0yBiDR4QRhh3DYIPNLVZckMAFF1A7wxBU/GGGFX8kXAIJOFcgQA5n/FCuyxpoUIILMxBhBiLSxDKAABZMUEIFVkwxBSXSuJKAAhiwIAQeqqg'
                    'CiyvJJCPNN+GYw80tTfTQxAcPS0DBAhH0J53FYwKgMZonPLhAYyJI/3DEEClYcQc37ABjhg5m/EBFCSlw4OcAHLTwcoNyXeCLN8P4IAC6TVuQgAYCJLCBGFdI00orv/hizjjmvOP6OtpQ4ww1knryhw0CYLAGHJCw8kw86TBjBQ5LpHZDAze8TRoB34'
                    'kAgJiB+UMmBQeg2QTETTC3GA5DuEBFBzzccYcTYlBCSSyUdK2+KggjEgmSvlBjDjnipCMNK5r7cHUHJtyBujTUCMc4xpGOcbSDHvBIoALpMQ983OOB81gHN5wRP1bAIh7rcEUOTBA1BkQGADdYVeSqB8Ln1e1XdusBBQDQhLsBYAZWcAEQzsAEIxSBCV'
                    'AwAhbEd4dEqCIWsRhGMf+eQURueCMd5ECiO9xBjnBQQxrxG4b81oHEcdQDHgesRzzgUY92tCOB7TjHOejhunfM4x74qMcV60GPdVBjDW8gAxXsMITwAKAATfDAA0BwBA+V8GIGmR4AsEMrjfUABiewwA+s8AIgYCGHYiCDFMiAhjJ0IQhK6EIXMkmHQUh'
                    'CEpCoRCUkYQtYwOIXYZNGwKghRFUWQ2zbaKMa40GPdMTjlmxEYC3DQY5xbGMbz5gGMophjWKwoAqxIIMYqECEpjVGYxToQQ0QcKsbmLBXaCqAjTigtu14YAlLmMELmAAGLBhhkljIAhoqCQg00GEOcaCDHgxRiVe8ohN1CEQnApH/T13oog99qIQevkC'
                    'HTfahFqQgaDJqIYlf/GJ101iHROmhDnM84xW2MIUpOmEIPXzCEn34ghoKkQc+OMIRWKCCGYbAGgCorQcNko41AVkQXznoBK/SzgSGQAQqiAELUsgCGdJJBkrKQQ5oYIMcCFEGL/ChErgQBiY6sYpXHOIThxCGMATBCGG8whSWMIQlZCELVFzCEsKoBi5'
                    'kQYo64OIZucCFLU6BDFwY4hOdiOsrSnEJWdjTFILYgyIG4QY5DPUMM4DPCZYQAwfcqjszrdsK99aAB7mpYzmAGQ6fIAUpVAELUZACGhKBBzmA4bTHNOkkHuEGS6ziEqjYxB5kcYk8/zDiEsK4hCFuuwphlIIRblhFNbqRjVeA4hXdqIYyOoqJaViiE7Uw'
                    'xSbykIZJiGIWtaADHQoxCDoAIg1zgAJiAcBNFgHgATGIS2SjR6YGLUCFaCrWCUaQgh/sAApjWIN+1wCFIARBCELIgjmNAGAhoKENXUiDItSwB0GowamocAML0uCGR+TBkiJ1Qxq6IAQ11MESpojqKT4hCVBYQg1dMAUjOJpgN3ABwXxQQhQKQeNC6KESe4'
                    'DCFnJggSP0oLIGwOkd6YbNyW4pLh44QQ0uZYUZRjK/Y4gyJasgBzywAQ1zaAOWDaGIOghDFl/ABS4u4Qg2VEEJczDEIGqhDFxU4v+dc5hDJQyRhi90QhjNEEYtdJHnRczBq3vl6yVk6wYlCKEQpABpHvZwCUWQgQkzwIEKI4CmGGTgjumlKUEEeYAtXedhN'
                    'UjBEqxww86SQcqVpEMZ5pCHNqShEIZYhCUWIYtOyCIbi8hFMypBCu3SoQ+dIAUoxJGNZHSClLWwBCnmLAlhe9UW5IgGLU5hjEbQYhW5GMUhHvGI316CEXoQRB32wGhGxOEMQ9hQBDJQgBs04QF3hME1MYamuFCghXuLwRG4RwUdnLMKVRgDGcpQBjSkgQ675'
                    'eeHn/uKXXxiEcbYxSksUQlLMKIPdGDEPUEhDF3g9RSjCMQiTKEMVOC1Dp3/kDgtcpELTBiDFrQYxSoacYhvM6IOl/gtIxTBaFQwQg47xsGFmNOEI9jqS/MOZJnu2GloxgCcMHMkJbVwYJPmgeeGKMUqtE6LV1A7GsY4BSbE8YpAyGIPjEj7KlChhlI8QhBVP'
                    'YVUP3HraAiDFoGw9ij2vgpMrMIYEacFzUFxCEbIghFidcNu+8oHMjgBBzW4UAEyEIZfSUfemh6IILdDK87XIAYzcEERoBAFLXShDYrYhNsPIYhBm+IRjDhFLoxRDXGIoxu0wEU3SsFVrua8Dh4GBSZkcYpqZEMYufjEK6KRjVwE4vmYwMTed7ELsBtD24ewJ'
                    'y5eoYc+GEIQltAt/5nZcIUU4KAx1nOeepNeUwCAwN6I4Q4FaoCDGdjACaRXApodYQq+sr7VlpRJlSALuoYLpbAIdCBmabAHeWAKoGAIImUJpfAKs4YLynB7tNAJp5AN0CAIgSAIICgLmJAL0VCCxhB9p3Btn8BPefB9lhBYjJAHWeAEQ3B+FBABEvIA1XMAkY'
                    'VNzvMgiOEmtIIDI1ACOgAFUjAGSsAFeeAIe6BhXQBwWkBwaaB/daZJeVAI72RbiiALdUAHaaBRGkUHdWAKpJAN3hAND7cMufAKn4AJ41YKoMB8xrAK2hYIh8B69BRYglUIC7gHfJAFRAB5kdcDbUIBD4ArmCdZBmAj8v+XZD2wBC1AAi7AA5xVBVSXBleoBVp'
                    'ABlWABFpACHxAcEowCHMwBoTAVGnwCr2QXYbwCs9gCWLWBaRgV6Rge6dwCoGAC9BgCXoACoxwCI2QCzAnjOJWB+MmCIYgW3pAXQqWB6ZlA5DXA9bBHTNCZPRGOQCwH34EApFohECABFWwTlmgBVWQBF+QBgAnSVR2Ukq4TlWQimNQBrjgDZOwXb0gCpXQDd6A'
                    'cMpQC11gDOKwC7qwCK8gDqagB5ugCIzwfIuwCKMgbiD4fYsGgATXBnzQeEVgA0sAXyjCHfDhUuy3afWGJaVxJQDwdEPwAi/wbwAHcJqYVFUQYGigTnIQBGX/UHqEEAfrNAaiMA1zQAeKgAt6IAroIA50QAreMA150AfC4A2SsAikoAyksHOMgAp6oAfP13oP'
                    'yJB7kAYINoVKgInjaAQ7YAV9VCGlEQMcMCcEoDGZJxCbByEeKQHhJENGgARYgATimAUvZmZoAHDqtE5jCXBxFphjYF1lAAiEIAqFMAvogA50sAno4A1lQAemkA55oAcaZQmWkHqMwGCkUArS1VF5oAZpYI4v+ZJZIGBUMDwLYCPSNF/whgB0w14liRgOM1mLVS'
                    '06wANYAFRAVQVZIIWVBHBc8IlCII7jqGVaoASEMAdaAAhlMAhlEAviUA+KIArdMA1lUAiiIA6K/1AIsmAKipAHeRBnV6dRC9kGVJiaVIcGXDCf6kQGRqADwwMgFGAZjOEAcYGNSrdCBxB/wXICTYADFdBIKmAEQQAFUMAEnWWcgakEUiAEWlOYY1lwWBYHqxZn'
                    'ZZAK3uANilAJvSALbRAHlfAMeQAIplAJ6NkGMHp1n6kICNYFWgCjXgCjbZCcnYUEOgQEkVYDH5ABD1CkiBEXi1hk8KE3+8kBINAEI3ABNqADKsADPGAEUBCcWHBqqykFPnpOeelZ41hUpxgFVTAHBLcJsjALX9CAm5AGcUAIm7Bq3VUGreYGrjYHepAGg0BdnAi'
                    'jWjCf8/mJSCAEOrQ4cXIYiPEADv/QAN7hPHEpPdk0oBSwHxTgAJ2GAyUQAh2wA1ZqBKDqo1LgX5zlpaDKBAOGqp5IZZUUBWh2pnPgCJMwB2pqCuCVCEy1ajqqo5iIBmOQB6aoiZpYBV6wTlyQBYXKAlJgBC/gAk6yANDxAI4qISOpeXYDfw/AHdwBAhaQAiFgAk'
                    'ywAkDAA0AAqnkZBFKgpUYABOXKrkAAoUYAoY5wimUAcBvAYSSVZpbwCL4qBVHgqmXQBnmABsVKBkqwapZ0YPrXnDmqZVwgBP/VWUBKAh0iAQAgMW9zsT1Ib5N1pN5xAuyhKSVgAmZZQ+XqoFEgBMuqQ2KwrkwABDrArjWUpXIgCij/1QWiEAuTEAdyUAZVNrAFN'
                    '3BxIAWiwAdz8JKeNZZaIAQ3igZJQHU7eqzFKQVaIwRgAAQuYAEwEh1oIm+DtIi46V5HWj15YyUa4AIxKwM1hENSAAUQ619BgKWn+rInC6FYAAZ4kAp4EAqc4Ay4egzHkAqzIApRUAZKMAbjiAaD4AgUql9REARmuk4YuaNjOahnRpwAlwRg8AKMMwGuYSOuIhdwK'
                    'Vnwl7GIsWRnuwEroALv+lPLGrGdtazlSrdAIANGMFTpOgdHBQyxMAuzkAihkAquSghkEAdx4AhI1QaiYApaUwVF1ZrjiKfHKQRKIJ9akASDOahYwLmNQwGOGgHx/6cdANp+K4QmicEd7rcuLmACVXqlWCAGSBAETxBwUbCl9im3PCADK5ClQEW1wSkGYpAKsZAKl'
                    'JAK1nAMcvAEZIC8WMYHxyAKBiZJnsUFB+YFAFcGXPC0cbCjWqBO5fiJzDoDKQACylEAIvAgDWAAIhmp04Mi1LOo2xgDKTA0G4ABK3CqwclZZOBfk0QGWQChbRsE7gtUQQAHEIsFpRULx+AM6CAP6AAO2wAMnBBwk8AHqWBoxRlamDifTxuwcuCcg6oE4siXUgCzIj'
                    'wCF0sB1oQY8JGk9PYrC+AbA5o3ElAD0OICrKsCevyu/YWuK4BOWJAIcmAEkYQFPHDDRrACWf9QC6mQCkyAB2AAvJ7ADvIADtiwDb87B2hACA28tIK5mkkghW2AVBRaBV4aBCwgjttLBDMwIgBAGSt0pAvgxgG6HSfgn3ujmxagAR3Aknq8xyuwAnELBTRABnggBT'
                    '2UQ+Zku3p8BsAwCYkwC2BwB4gAvMRAydaQCMPgCmNQWoWJBqlgyjzKtMmJuVVwrM9pynzJl1jABETgAikwAochApQGhASQKyaEm++3AJ+xpA3gASAwAryMxxjwAr+sAi9gpcv6vkaAt1DAAoaMATKgAjpACIiwAixACcdACVggB5QQCtbADsWQCsMwDHHgiUvYBl'
                    'tDCOnqw1JAwX2ZBWLao3v/uc5AUAQ6QC01sJ/Usx0eS8vtJwIEACEegzcgwMtDw5IvsAEvELMvsAOsO66EjAXiWkM78AIYoAPlRAlksAIHPAuCCw7gIA/O4AyhsAY7DLFaIAeOwAlyMKprwAVgcGCfFZzp2rbBGa9AEMzN2gEVYAEPEwEF8DZKRj3tVq1yWW8Dehg3'
                    'Ar4cEDQl8ALv7AKU/QI/YNAx27pYoAIywK4ywJImwNR4EAtgsAFskAio7QjEAAzoAAyhkAj++gRPIARVwAnHwAmkqsN4cMxAELfxmtfmGsyf7QIloAEQcBjV8zYgQCLxhtiSSjn7qRxxkQER8AAWQAIlIE7ETdncrQJU/wCzN60CNlxDL8C+LrABGwAEkCwDGOAI+'
                    'jgIX1AIx+AK8EQGT7AGqIwFopAKWSDMUPAERpAFSGVOVtrZTGCytSsDn92sxV0eAGIj5tsl40uS5UsBHHAALowcBzABMzw0HVACQ+MCHdAB3L2+/MOSTKADOrDUJrCpG6C2THBMhGAKotBdpiAEJ42uZKCsxIkG/uW+DIoEf1wEE70CSg2zCq7iMEvcztIxzHEAQ80B'
                    'KVwAo4tNvxIgD7Mx/UEB6zLiHaAAI04CIx4CZP6tJmACKq7iB77UHbAB540BrIuqaDAJ6VkGpjAHwsygbZtD8bqyBz5gLIDQBs2+FF3Qkt3UK//uMvjBHRhOXhn7R3Wjfvz8NhyzABMg0F4+4tVS5iFA2WZuAi7wAjfETCuu1C5Q0EWgQ1WQnlrQahAtxObKAuWq4Fdq'
                    'rqgKszoQ2krN1Cqu1M0qwk1jAA0AvgvA2He0sUpHOZb1MIkxARZQASMu4sSdAy4eAjawvmR+5iZABcwk6kXQrOXt5izJA0FQBYZrjg1truoursG8Albaui/r61jN1PL+Ah2gAZZyJf6BU3mD4Sbs3Ju37G8D0M6uARsA6pRdAjZgAhu0QSP76QhNBKVe3pSN8KFepbRt'
                    '7njtruPKruIqA+TK8UBA0b3erEwd6li9viKO7+qhR1SOU/65HED/TeHekSJJVqknICQDXfHc/ek+MLLa3utFgLXri/AgDuI1rAKchQaSFK87QKWNJPLlakP5O+g6IPF4rNQmANoYMOLFrSLOcQDO4QEAcuwAX5KxGU20chgTgNTlDe5Ff+YljragrgM4ffBnbgMlwD8g'
                    '3unvGgRZwF8p3tRA0EiZreTgveIqjrZXz5I6EOrbTdlergHyDCHXo63/efYUEJLR9DDDEtDlcuZuHvfs6wNzf+YxG+o2wD+5TuLEXS1om+pPkKX9RvFvL/FbX+qL7wILT9m9//jc3eKc2gEXcB8OYxoPsx8qDLbZGBdz/DYWzgG7zD+63qza7gKmX9m930gj//vOoK73Q'
                    '/MDP8DdHD/0Kn79Sj2lj6/toC73nh4CIb7dXv408sEBHiAClZVkM4LPLPzKGwMQFBYsoHAAxIkRGjSUMGHihQuIEU246DBRh44XOoq4mOjiwoUcExmWKPHDxY4dOnhcdGHDRIiKHWRylBkCYoePOS9UdFGiZ4cSMnFWuDDihAgKDQaeWAAAgAEYAPL5o1q16jUAFJwSP'
                    'EDQwYkTMSxoqBjChMwSIczC9EnEZcOXM3PArLhhpg0XQFDu9UERbwcXVn7MpJgAqFsrVKwQKXEBhVCcOD9WsDAiBtIIWh08IPA06lSrVrFqBUCQwmnUI8ZuMGsXaFq1cTdsMP9Bxc6ZM1Q8lpjrgohiK8F/89ihAqUKoC5g+v5xISjQH8KBEwk+OPJOnRUgWOAQIUIGCgg'
                    'oPGha4HNo0VmdZq2hNSmFESM+CiXpM4dPiIntUKFihgqRO+zwKDoqzjAjOCty4w8KHlSA6wyKINoJoh8InCLCEHIKzAqdsqugghhOoCCzBWLoAQECCiggBqnQu0q9rE5rqisKQFBtrAvYCsEGnNKy4bfFXNhAAcDs2MKFMxA544qWhHSBii1wA2JKHWqz4QUiiPDNjjsE+'
                    'ymBxujayQYOP0IhJws+rMAogQ4A4AEOTnOqhxb7cRErEJySQMQCSjsNBLHSJAsm2mSiAq//+UzwwSwfEtCBiSTP0IgKjKgAwgUmChTjDiaAUEGFF0Il4ow7EKHOjgsEOKsDLX+wwaQMP8rhzAxR+HACEESIoIE+D4izKQDOc9EfrEQAFiz3JKhRNQ1I2EktoVywg7GPdjSB'
                    'Idxy08GKEBQ4ibYtmDBBhx2MEAOMM5jI1AUdCmTCP5/MOMOOITT4rb/gKJIpB6FiJSoFZL0DQIQamiLggBNaHHa0pw5A7WEQOJigAmcvmK21EEb9794t+OuYIxNWojS2cS8iVwUg1OWviHudFFIB2+wwQ0CX0uqXLpl2ataCCeLMrIEaHegMYYXvBIADALoayAHUAFWNhJ'
                    'xZ24Aj/7t0GBU3/nSwYbaMqjShiCu2LrlkHXxISYfZdKBIAYYsqqsEG56zqSebO4DpbpxI0MAoEUl0YAEEnLqhaPQYdkrGgZaVeKegGqINoodceAGuhj7VwUEgTNjhJR8sn1KvF6iEK3KOIBp33Bc60CGEh65tqLW5QuCXLRIsAFREghaYYIHOABgcNKM9WI+CHgqiIAMOO'
                    'LAgzeQcAjVUIC7q+iK9PkV5ypSMMzslIz7vnrjNIWKJXZawZAmva0t/aSL2R+pAOwpOyGCBAsJyYKuEp7KzcACEP/wED8QvBhKbwFjs5rxQYYkIF9nB9KJ3uc8RxwnkSsmUumeE7+2lISCbSP9GHpIRIoQKfRvED0TwIrm4VKBnHhDBUUCwOzfVj3ChwQrSBjKQHhRvBMl'
                    'bHgnwI72LFKEIVPqcDoBghCFi8HPR854RsIBBJGCwOJ8qXUQi0jW9dKQjKISc2qTHI8qM4DQn6AEIEDAQwc0wPaQxDVgQwMOxkERyoVIBRjQChCECIXQyOKIRmIDB7wGBCt1jARBYgAQsSAELWBADFjwFMvFdxCE7oBJG3iIRyc0xVBhRjq0AJZCwCGRGIlDji/y3AA+I6H4'
                    'RW54BaQMq64WKj0CQwQtWcEQLAoEHQLglDzD4RyMgMgtVyAIY2EAGKDTQBI8zWcnwWD0OTuSECUwgR0L/oJ0JDPBoPYiBBLZCSuDxT3g3zKFATlPACvgEgaBySEamVEsZ+LGPRlhBPY0QTyYgYQVGEEIW/IkGMGTBCEEIAkZYwy6O1JEK6prSp/rCLg9qsiEYWOYFKDMBZcW'
                    'vBiAA1gKEZTTSPEyUQBvBDHxCzSK8IJYqfYEM+ChPJdLylrrs3hHJEFAueKEKLAiCHzFCUQ+GsAhMGKJKN9iREy5TqbPpgELEuIAIZPQEHOCdeUpJlRp2Jn73S1oDDiCBCZRgBl2rZUod9CkMgIqJFlQXIAnpx07xAAtk4AMftKAELVRBCt0rggycxMmhDpGiVYSLTXjSuJ9'
                    'oYHcLOECfYuSm/2BdlVhZYayMCtCVDCgFAie9SC0xooIVxFIG9TwiD3YJBUAmUghGkAIUnhCEnQriEV6wqxbQgMGfLpOi41ooL6noAhVAJLgVcd1r4nYBBzhgVw0wQIzKQ6dw0hBGFPDAQLqyqwVYAG4qdakMirDL7s2UBfvcZyKhGAQkIAENQgiCFMawhiCM9xGo0MIe2pC'
                    'HKigxgSpgprr+SKU6ek1IJYhcBx5SAg1UwAESUMplCUJVABTgd8OarFbEA4KCIOAA3gHBWEygAgyMdp+/TGR6g6lIKWRBCkgQAnuzoIVJaIEFQhjDGNCABizMYRMw3gQn2mBPI6BtAxjAwAZeMFSi6v+FiUYsDgghMgPf9ESxcXKTVz0Aggeo6KPi9BPu7hcBEajGYqC1JxMWiYUqV'
                    'IG9sEUDGdCABL1KIQhRaEMsHKGF1VYBDXGQgyNS0YsvJGISsxAFGvZJCUqsQAF0RBkwV8ADJkBBCp16oB4nggG1xY0EOxRIVg7ggRhQQEUsiu4aYyRqDcfPRj4xYk9ZcOIVpxkNZWgxGuSAhiooAc5SQIMoZsEGObOWDHHgxCyOAQ5gAAMPk+BEIoLAhETMghI6cClou+dSI5BBD'
                    'K1trRFQdrl2crIxFghRUpqyAA4Iz6r6M1qeFsCU0jQglRwoaV6w0N70whnOVeDCjdtgWznIQQj/ec0rJxwxCDakNwpiQAMlZmGNbcjDGtZIRSxmAY5jbGASieA4BlYwWiwwgaBgwAIPgoBa0x7xp9R8iAaWF6cbeDMrJygAVCSLlYQtQARaNecIcGADHZxY10JAghTSXIYbz+HWbZ'
                    'jDev8tBSGIQtZGyIKNUwEMa2BdHdggRiQWeYxjIAIMQgAGGFYAhfJmAZHOjkLA731E1HoqgZx8H6dl9FVR27zUL7KwCBx2GgIsoKQ2KIIRnpBmo6dZC0zXwhwG4QUllOHfaNACnIUwCE6gweiTcAQbQlHxWGBDHuUARizkMNdNxAIPs5gFE6QABnoWnQVgmISeb83aJ0a6rQ2i/2'
                    'IHSIADjvZPAgtQFgAasGXpIu3vpzkAAQhQ0g4AIfFcoD4XbNuG2rbBDbnmAh8oLwQuqNkRaGg6H+bQBk4cY+LskIczhhGLRODBzZMoBifK+0TcIwEMeMBDFtAwTEWKIihYJAb5FBOIG5dzgK5QFgmQgAcwPuQzNefSCglAABwICiAQA37jgjbANS0ogzTQgrtavBsrAz1rA/DLAiHA'
                    'AEc4uDnoAjUQBawDB3mQOGtwhlDgA1obAylYgXsjAyxAglfjJ70CAy4gA/TKAiQ8OQLkveHaG46qLBmBwJvrn/8BHFRygHozMzgDA2LKNSXogrziN6a7sTRDQilogxUDO/85mIM5mAQ+aIMvSAX1mwVCmIXSE4MxiIIxyIL2ajN6UoEhVCQ3WzEsUDsjQK1Jq6Rr0YCNGhEDWAAHgDA'
                    'DyB9/2B/p8h9l6YwF4CgIIAEbUJdDSoJ+SjO8CsEqWLw26IIuGIMgeC8oIAMlaINiCIUq4LhJaIM4GIRBSIQ7TAVOKIZimAVfIITx44E4i4IgeLSTkzMpIAMpiEaiMy8/YhlJKoE1YZoM2AoO6BO9ozDDIZHE6cQU+AF1YTEkKCY904I0QMUuQIM2KAMtGIMyAEIyIIPFEwVHqII4UD'
                    'MhKANH2IRjMLZZiLZZcIRmgwVOQLMVIwM/NC3UgjWoiwIhCML/C2KCWhqwCkiB93CYB8iTCLPEhbFCAGCa4RMICQCBGfiBoUJHJSxBEZy1MpiDPDi/G4uCKpADNmgDRRAFRdgzyiuDmayEQeCDNIgDNEiEVHAEKYgCWJgFaGSDKgADMNgrDOqpJyi6aKwCKIiigXqnyVELElgCDmgAp'
                    'RgIDnCTSqzCgiiA3IkfEakBHBgCLkwvQiQDeewCvATBNKCDLzBBPpMD7NuDPCiDMXA8odSxSkiDSaBJPlACJagxnAy4ORimLFgkRUKtKIoCo0s8o8MC1KqnIqgS3xsCEPAA8GgAAqAAA2jNCLyK1SwAAgAaqDoKMhqCHKCCIFQCo1NCNoBH/6TLyzTQgzzIgzYYhPFLgzlQAz1QB'
                    'MfbhDaggzkwBVOYA0eohU14BEf4SZx0s/9rsyB0SKILpqJLM8TrTK+sqZ5AgSVAltNwgFOasJHUCnQbo4RJySGwAia4xzQYJu+MSRPsgjLYg0OwhE0oBOd0w+I0BVngA0uYBTqgAz7YBEfgg0E4hldQhOKUR2JSAllDg35LMSk4pCpIrzQLP8TbN4u8FLFaAglgGhGIkxE4N+i6R'
                    'KNRPuoiI9LIgPw8AzCgvFyzMXxMA6ZLgy/IA0GwBFC4BJrMA0WoBEvYA1CwhEJAhV4ohC4ohGrAhQkVB1nYBMdjvEGYg39EA0JgAzlgsf+dYoES3bcqQDoQbdNJ654csIEpqIE+WYAbAIvgM4Aanc+miAAy6oHqAoARGIIfMAIw0AJWhMxUXLw0UIRC0ANBAIVS+IRLqINBUIRN'
                    'sIRKYIRSWARGEIZuuIQ04ANxqIYv6AV5mIZNGARG4IPC9Ml/LINbwzXF86dolAJHjce8UoIkiEYW4AEX+IEhCDX6ISNkLQ2RBCmncJgbQpgmGAInyLFBSAM1IFKmo0lFsARFMIRDOIVVMAVMMATC9FZDkAVhKFBhEAdSKARTSIdqKARW9QZFmAVk8NY8OIZNmMUqaLo2aIMqwL6AB'
                    'UDzvCsluDEzdCS3mIElCDXZXIAHaC7/42tW/mEjxhIPCqiBIxgCKgADPniENmjHJuUDRWAERhAElcUETBgFTLgERjCETtUDXBCGT9ADUi2FdO2GaSgES4iHbugCUcCFA92EaTCFLkgDM3VDgIVDOVA7fvMCEdQCL6A8EPUnFv29I4CB+XlRBGguGdo7rLJCSRwjbkKIGaACNuCDP'
                    'VADNXADPWjHQdgDTk3ZQ2DZRqgDQagDOjAEQ6CDbBCHRaiDXKgGd7WEZugFOpgEcXiGNKgFUlCEL5CFagDTLJ0D6ZwDoRSFQbi1VEQDL2BFLwjYgEU8MdCBEkgBExnUABqPp7BY6dIKhxGRP+mBEUiQOdgDQ9DbQNDd/90VBJQlhUAQBJalhVzYBUwIhEVYhEAQhmwIBFs4BWioB2'
                    '/4hF1ohkWwhGkgBVxQBkXQgy8QBmSoBT0ohEKohEIYhEJoR5MdBFxjOqEMwTTIg0HoQL0ig0OZgRQAgRqgXbBoigaA3fRAmtLogYQBgB6ogdtlgjmwhEOog0AIBD3wW+I9BVBghDqoA1qghWiIBmjYBV1Y3kM4BGFoBlvohF0QB3Tohl14hU74hFyQBWRQBL8khVf4hEpYBFD4Al'
                    'Kggzrw2zpozoDdg7j9Aua0BFPYBOfkUCnYiBnAgRq4gQgAAAm4gRpwigCuE6M5YASmE+JpAhz4gTOIA1IAhU9QUv+/NQRBKAVQWIVSaNld6GBj+IRXwIVaWIRKAIVXEAdokARJMNxuqAZbAAVcaAZcMAUftoRVEIRDCARQgIZVqINSeAVSjQZMqAO33QM3cANGsARGuIQk3gM6EE'
                    'O9opQfgOImMBYJSGWnMABSA0f1GIgmoJNUAuPEAARQNuNPMARONQWYPYRSGAVjMIZoMIZdGIVOsIVagNJKCIRoEIdcCIRVyIVcgAZcOIVcKIU6GGFJ3gNvLgVBaAZx+ARjEIduyIZoGIVdYIT5QoVVSFm91eRsFUG9MgIiMFYcaIIMaMAwyDkAeOWRJOAaaAL34ADblRdC2ARQCAQ1'
                    '8NtNwM5SEFf/TCgFYzbmUTgFCwYFZJCFPjiFV4iGbJAFQZDmUniEVxiFRliFT2AEWTAFRWiDS9gDRvgEaHheXIiGXTBmUEAFUWjnR/BmQyhMvKK+OHMCIpiBGqiBBlyAHHrAyBLbCnOKJsi5PeGmmQGETdjbhubUQVCCNABXMzaGV8jpdL4EYTAGWciFVQiETkAGXJgEQ/i3NHiErO4'
                    'FXKADU2iGPJgEIUCDrzaET6gGXeiDXdgFUMgFc521vc7FKqBHNBiDRLqDNYhGIPiBJ+6BBiwA4hmn1xxb0mgCFjmNWeYAK/hRRvDmPTBQRYi8L/iCPZiDPgCFPmjZXMDkUiiFL2iEXMCE/z6gAzgAhEpQAhYQTDRQhGeoBThQgj4oAynAgDDsgnc1BVz4Aj2wBD0I4rwaKPZSxifgA'
                    'ShYODFYg/CGAqupjCZQagRAgP4tYLb0k3EqpwGwgjsobjXwZka4tUnABVD43i+oVEz4hMO2hD54hV6wBEuAhmiohAOthWmYhU3oZFNQ0maYYQSlg0F410qIYSQGhet+hEdwAzwTApxk2jWAAihYAzAQAxU3cR34gRRgADIaPmh1k/PAxPQYJ6cAiyPQCgOQFyxgWzegW1GYBFEQhVI'
                    'QhUsohVx4hVPoBEsQBkMohFowB29IBmloBnJohlIIBA8/heZdBUxghFWYBlkgTP9FgIZm+IRT+IQyLgUwZ+lLcAOmRbo5kHOFJYMqOLMzON0ZGIEcN2ClcIp1g2XhcRMyaoIFMAARgIEl+Fg5yANGqF9HIIQXtASVnWlQkAVc+IVnWIdzoAd8qId2eId2SAd0IIdneAZpcOFPGNw++I'
                    'JNcOk9KARkMIU+cHJbsIU6SGZLcARGyFY3fNPzi4MOpDyoy0kxsIIcGIIjKJ4C6AHQ1gq37OyodhiCTpgbgIEYFeM1iANFmPQniIJtSAdv8IZ0UAd0UAd1IHd6aHd3b4d2gId3iId6CPV4uPdwkIZvQPVn2AZtGIdiGIdwGPhwSIZlMPhlqAVb+IVKIARIuIP/bYs8NIBM98oCSRMDIg'
                    'iBKcABDjiBG8gsGAgLtRRg2PQfZEHgG+gTAkABKniCbocEFnCFcwCHdUCHdagHUMeHeKCHecAHnocHc4CHUY/3eJB3cxiHdGiHgR8Hc3gGalCFVvgFWPiFcHAHcnCHqrf6gdd3aSgGaRgGX4iFZkM0iH+CgaKkV1mCCeiTBkB0AII332lvADiIE1jlhNk5ErACG7iCNxCCYoiHdZAEOG'
                    'AFVmiFVsCGczgHdmAHnqeHdXB3eoAHyR/1ox+HX/iFVrCDDhCAzhAAAagAAXiDYBiHdTAHZnAFO3iDHTAbH7iCUbGDSIiEO4gEVaAESDgDKEAE/1Wwgk9sggmQE0KNn+LJ8Sq0ofgxJ/AItSWYgRzYASjghnSIB2mwBVYgBFY4gx1IJzMIgztQhUwIhmEwB2qghm9wBmoIhlZQBTtAhF'
                    'twLc/vgBwgAfkvAfn3/FvAhj9oLs4XACL4A5MCiCVDNAggmEBAghw67Dijh83TkiYnJIA4UUNCgRMnDgA4cAJAPn8iR468BoACgAUnKSxAIOLEAwo4juSw4QqbNm3PpD0r9stVsmSsIMF5guFJsmGUKjCY0WIAgKgJfBGjdIdKjhIpJqRIMeOHhhRDlkwAYAEFiQoXrHj6M2VJ2x8zhky'
                    'ZoaECiimUqN3y5ClMjxEgKCBYsP+AJYACB26AJOnYJEqVC06w1HhDRJMmOChMQUHFzp1IqlS58sWT2i9IrH55S6eUCgkBS2ZYIGGBYatIM0poIEFixIQRKS6MtUCAAA4cEAgwgGBhiJUwEwQEZuBgAgQEzTlGrTAkTBgQDmJQpnBCZUrGIfs5Fgk56uEa5xFQaLKgSY8aPeqjhFB2wg'
                    'xmiObLL9RwM84zvqyBSAkVtDACcAxU4MktQ5hhAQQZcjABAQA4MMAEIY7AAAMLWEdiVARwcIBhBRRwGIwUUEDRCEdYNJGMEpi3UQFRwdBYe+4t4EFUL1JQQwweMRZBBj30AF4TNXBAwYgABHAlCFPcckskb6D/gBdzlEmwAAQM1DACABMgcFyHiankwAIOTGmAAxRwwMGaBDgAwmEUNHD'
                    'DDSwVEMEB5lV0gpMxnNRDoRpFkJJKMVAQUpD+mNRDilGZJ4EEhUVAQQwi9JBBBA0YYGKdMIiQAQwZRBUVAok50MABpUYwZVR7krjAAy0uwKJhFGTAEgVwHgCsncOa12oDzS7QwLAniCqBpHXuSAFHkmFKaZD5AICojIYZ5gCnM876QAMPZJAuBRGUy+kCOlIwmIzCDqtsvfn6WWq7/DYb'
                    'waftytjAs/zOOKO8+j7wQAQx6kssBR40AcA1/FTKnj2vwvfqcVFB5aLGABgQ8qs9JpaYiz2quWwyyCuDnFKmJ6NHssYm10xzkfFG1UBUvFhc6Uj5lJPJAgTA+CakkGbwwIuSFc0mARQwwNIAFFwwAAoDQAWzxjIuLOPSKMGnktG8GnZSAW3C3ObZ4QKw8GEp+UlBTB4YwIs9+wAtEnsi7a'
                    'PP3/oIvg/hgxdeuOCGEx544o0PDvjiiCMOOOR/L/445ZFbHjnlilv++OY/+9P3SKSLJPreqau+Ouutu/467LHLPjvtjvVteu2567577gEBADs=',
        'blank': 'R0lGODdhdgB6AMQAAAAAAA0NDRoaGkBAQGBgYGtra5OTk56enqampqinqKqqqry8vMTExMzMzNLS0tzc3ODg4Ozs7PPz8////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAkKABQAIf8LSUNDUkdCRzEwMTL/AAAMSExpbm8CEAAAbW50clJHQiBYWVog'
                'B84AAgAJAAYAMQAAYWNzcE1TRlQAAAAASUVDIHNSR0IAAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1IUCAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARY3BydAAAAVAAAAAzZGVzYwAAAYQAAABsd3RwdAAAAfAAAAAUYmtwdAAAAgQAAAAUclh'
                'ZWgAAAhgAAAAUZ1hZWgAAAiwAAAAUYlhZWgAAAkAAAAAUZG1uZAAAAlQAAABwZG1kZAAAAsQAAACIdnVlZAAAA0wAAACGdmll/3cAAAPUAAAAJGx1bWkAAAP4AAAAFG1lYXMAAAQMAAAAJHRlY2gAAAQwAAAADHJUUkMAAAQ8AAAIDGdUUkMAAAQ8AAAIDGJUUkMAAAQ8AAAIDHRleH'
                'QAAAAAQ29weXJpZ2h0IChjKSAxOTk4IEhld2xldHQtUGFja2FyZCBDb21wYW55AABkZXNjAAAAAAAAABJzUkdCIElFQzYxOTY2LTIuMQAAAAAAAAAAAAAAEnNSR0IgSUVDNjE5NjYtMi4xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABYWV'
                'ogAAAAAAAA81EAAf8AAAABFsxYWVogAAAAAAAAAAAAAAAAAAAAAFhZWiAAAAAAAABvogAAOPUAAAOQWFlaIAAAAAAAAGKZAAC3hQAAGNpYWVogAAAAAAAAJKAAAA+EAAC2z2Rlc2MAAAAAAAAAFklFQyBodHRwOi8vd3d3LmllYy5jaAAAAAAAAAAAAAAAFklFQyBodHRwOi8vd3d3Lml'
                'lYy5jaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABkZXNjAAAAAAAAAC5JRUMgNjE5NjYtMi4xIERlZmF1bHQgUkdCIGNvbG91ciBzcGFjZSAtIHNSR0L/AAAAAAAAAAAAAAAuSUVDIDYxOTY2LTIuMSBEZWZhdWx0IFJHQiBjb2xvdXIgc3BhY2UgL'
                'SBzUkdCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGRlc2MAAAAAAAAALFJlZmVyZW5jZSBWaWV3aW5nIENvbmRpdGlvbiBpbiBJRUM2MTk2Ni0yLjEAAAAAAAAAAAAAACxSZWZlcmVuY2UgVmlld2luZyBDb25kaXRpb24gaW4gSUVDNjE5NjYtMi4xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
                'B2aWV3AAAAAAATpP4AFF8uABDPFAAD7cwABBMLAANcngAAAAFYWVog/wAAAAAATAlWAFAAAABXH+dtZWFzAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAACjwAAAAJzaWcgAAAAAENSVCBjdXJ2AAAAAAAABAAAAAAFAAoADwAUABkAHgAjACgALQAyADcAOwBAAEUASgBPAFQAWQBeAGMAaA'
                'BtAHIAdwB8AIEAhgCLAJAAlQCaAJ8ApACpAK4AsgC3ALwAwQDGAMsA0ADVANsA4ADlAOsA8AD2APsBAQEHAQ0BEwEZAR8BJQErATIBOAE+AUUBTAFSAVkBYAFnAW4BdQF8AYMBiwGSAZoBoQGpAbEBuQHBAckB0QHZAeEB6QHyAfoCAwIMAv8UAh0CJgIvAjgCQQJLAlQCXQJnAnECegKEAo'
                '4CmAKiAqwCtgLBAssC1QLgAusC9QMAAwsDFgMhAy0DOANDA08DWgNmA3IDfgOKA5YDogOuA7oDxwPTA+AD7AP5BAYEEwQgBC0EOwRIBFUEYwRxBH4EjASaBKgEtgTEBNME4QTwBP4FDQUcBSsFOgVJBVgFZwV3BYYFlgWmBbUFxQXVBeUF9gYGBhYGJwY3BkgGWQZqBnsGjAadBq8GwAbRBuMG9'
                'QcHBxkHKwc9B08HYQd0B4YHmQesB78H0gflB/gICwgfCDIIRghaCG4IggiWCKoIvgjSCOcI+wkQCSUJOglPCWT/CXkJjwmkCboJzwnlCfsKEQonCj0KVApqCoEKmAquCsUK3ArzCwsLIgs5C1ELaQuAC5gLsAvIC+EL+QwSDCoMQwxcDHUMjgynDMAM2QzzDQ0NJg1ADVoNdA2ODakNww3eDfgOEw'
                '4uDkkOZA5/DpsOtg7SDu4PCQ8lD0EPXg96D5YPsw/PD+wQCRAmEEMQYRB+EJsQuRDXEPURExExEU8RbRGMEaoRyRHoEgcSJhJFEmQShBKjEsMS4xMDEyMTQxNjE4MTpBPFE+UUBhQnFEkUahSLFK0UzhTwFRIVNBVWFXgVmxW9FeAWAxYmFkkWbBaPFrIW1hb6Fx0XQRdlF4kX/64X0hf3GBsYQBh'
                'lGIoYrxjVGPoZIBlFGWsZkRm3Gd0aBBoqGlEadxqeGsUa7BsUGzsbYxuKG7Ib2hwCHCocUhx7HKMczBz1HR4dRx1wHZkdwx3sHhYeQB5qHpQevh7pHxMfPh9pH5Qfvx/qIBUgQSBsIJggxCDwIRwhSCF1IaEhziH7IiciVSKCIq8i3SMKIzgjZiOUI8Ij8CQfJE0kfCSrJNolCSU4JWgllyXHJfcmJyZXJo'
                'cmtyboJxgnSSd6J6sn3CgNKD8ocSiiKNQpBik4KWspnSnQKgIqNSpoKpsqzysCKzYraSudK9EsBSw5LG4soizXLQwtQS12Last4f8uFi5MLoIuty7uLyQvWi+RL8cv/jA1MGwwpDDbMRIxSjGCMbox8jIqMmMymzLUMw0zRjN/M7gz8TQrNGU0njTYNRM1TTWHNcI1/TY3NnI2rjbpNyQ3YDecN9c4FDhQOIw4yD'
                'kFOUI5fzm8Ofk6Njp0OrI67zstO2s7qjvoPCc8ZTykPOM9Ij1hPaE94D4gPmA+oD7gPyE/YT+iP+JAI0BkQKZA50EpQWpBrEHuQjBCckK1QvdDOkN9Q8BEA0RHRIpEzkUSRVVFmkXeRiJGZ0arRvBHNUd7R8BIBUhLSJFI10kdSWNJqUnwSjdKfUrESwxLU0uaS+JMKkxyTLpNAk3/Sk2TTdxOJU5uTrdPAE9'
                'JT5NP3VAnUHFQu1EGUVBRm1HmUjFSfFLHUxNTX1OqU/ZUQlSPVNtVKFV1VcJWD1ZcVqlW91dEV5JX4FgvWH1Yy1kaWWlZuFoHWlZaplr1W0VblVvlXDVchlzWXSddeF3JXhpebF69Xw9fYV+zYAVgV2CqYPxhT2GiYfViSWKcYvBjQ2OXY+tkQGSUZOllPWWSZedmPWaSZuhnPWeTZ+loP2iWaOxpQ2maafFqSG'
                'qfavdrT2una/9sV2yvbQhtYG25bhJua27Ebx5veG/RcCtwhnDgcTpxlXHwcktypnMBc11zuHQUdHB0zHUodYV14XY+/3abdvh3VnezeBF4bnjMeSp5iXnnekZ6pXsEe2N7wnwhfIF84X1BfaF+AX5ifsJ/I3+Ef+WAR4CogQqBa4HNgjCCkoL0g1eDuoQdhICE44VHhauGDoZyhteHO4efiASIaYjOiTOJmYn+imS'
                'Kyoswi5aL/IxjjMqNMY2Yjf+OZo7OjzaPnpAGkG6Q1pE/kaiSEZJ6kuOTTZO2lCCUipT0lV+VyZY0lp+XCpd1l+CYTJi4mSSZkJn8mmia1ZtCm6+cHJyJnPedZJ3SnkCerp8dn4uf+qBpoNihR6G2oiailqMGo3aj5qRWpMelOKWpphqmi6b9p26n4KhSqMSpN6mpqv8cqo+rAqt1q+msXKzQrUStuK4trqGvFq+'
                'LsACwdbDqsWCx1rJLssKzOLOutCW0nLUTtYq2AbZ5tvC3aLfguFm40blKucK6O7q1uy67p7whvJu9Fb2Pvgq+hL7/v3q/9cBwwOzBZ8Hjwl/C28NYw9TEUcTOxUvFyMZGxsPHQce/yD3IvMk6ybnKOMq3yzbLtsw1zLXNNc21zjbOts83z7jQOdC60TzRvtI/0sHTRNPG1EnUy9VO1dHWVdbY11zX4Nhk2OjZbN'
                'nx2nba+9uA3AXcit0Q3ZbeHN6i3ynfr+A24L3hROHM4lPi2+Nj4+vkc+T85YTmDeaW5x/nqegy6LxU6Ubp0Opb6uXrcOv77IbtEe2c7ijutO9A78zwWPDl8XLx//KM8xnzp/Q09ML1UPXe9m32+/eK+Bn4qPk4+cf6V/rn+3f8B/yY/Sn9uv5L/tz/bf//ACwAAAAAdgB6AAAF/yAligAQlGiqrmzrvnAs'
                'xycw3qRNGUjv/8CgcEgsGo9IoiGHG+kokYl0Sq1ar9isdsvtZiHO5pPC8JrP6LR6urjpxgvHek6vpx1tp44hsfv/gFYSDEwUD4GIiXUPhXKKj5BdDo2RlZZVkxQlIo6XnpCZmxSdn6WBoTqkpqt1qJyssK2UsbR3s7W4kre5vFiuo73BV7+qwsHExskTyMrCzM29z9C50tO11dax2Nms29ym3t+f4eKX5OWV5+igu'
                '+vj7e7m8PHp8/Tsmqn37/mv+/L9gP2zpG6gnYIG6SBMuGYhQ1sBiz3843CimYoWdUXMiAgjRy0eP/qyJxKiKIklTfvqSylrI0uFJF9qPClzTsiaN2XmfLmTZc+UP0sGFTn0Y1GORzMmtbh0YtOHTxlGTTjVYNWBV/9l3bf1Xld6X+OFdTd2XVl0Z8ulFbf2W1tub7PFtTZ3Wl1od5vlVbY3WV9jf53FrAlyMOGRLg+'
                'fCXzMsGJMjh9PYRwtsuRlliVT5rWZWubHnXGFvvZZ8Whap7WVPpwaVutuqwm/XjUbXGychQ5d7sIoIAIHUXZjieAAQaERfIRTidAAx5sbCZRPOeBGzIgGELJr3869u/fv4MOLH08efHMSTY6nX8++vfv38OPLdzPGuQkB+PPr38+/v///AAYo4IAB1oBDCAA7',
        'carddirs': [
            os.path.join(rootdir, 'resources/images/series1'),
            os.path.join(rootdir, 'resources/images/series2'),
            os.path.join(rootdir, 'resources/images/series3'),
        ],
    }
    # 音乐路径
    AUDIOPATHS = {
        'score': os.path.join(rootdir, 'resources/audios/score.wav'),
        'bgm': os.path.join(rootdir, 'resources/audios/bgm.mp3'),
    }


'''记忆翻牌小游戏'''
class FlipCardByMemoryGame():
    game_type = 'flipcardbymemory'
    def __init__(self):
        self.cfg = Config
        cfg = self.cfg
        # 播放背景音乐
        self.playbgm()
        # 载入得分后响起的音乐
        self.score_sound = pygame.mixer.Sound(cfg.AUDIOPATHS['score'])
        self.score_sound.set_volume(1)
        # 卡片图片路径
        self.card_dir = random.choice(cfg.IMAGEPATHS['carddirs'])
        # 主界面句柄
        self.root = Tk()
        self.root.wm_title('Flip Card by Memory —— Charles的皮卡丘')
        # 游戏界面中的卡片字典
        self.game_matrix = {}
        # 背景图像
        self.blank_image = PhotoImage(data=cfg.IMAGEPATHS['blank'])
        # 卡片背面
        self.cards_back_image = PhotoImage(data=cfg.IMAGEPATHS['cards_back'])
        # 所有卡片的索引
        cards_list = list(range(8)) + list(range(8))
        random.shuffle(cards_list)
        # 在界面上显示所有卡片的背面
        for r in range(4):
            for c in range(4):
                position = f'{r}_{c}'
                self.game_matrix[position] = Label(self.root, image=self.cards_back_image)
                self.game_matrix[position].back_image = self.cards_back_image
                self.game_matrix[position].file = str(cards_list[r * 4 + c])
                self.game_matrix[position].show = False
                self.game_matrix[position].bind('<Button-1>', self.clickcallback)
                self.game_matrix[position].grid(row=r, column=c)
        # 已经显示正面的卡片
        self.shown_cards = []
        # 场上存在的卡片数量
        self.num_existing_cards = len(cards_list)
        # 显示游戏剩余时间
        self.num_seconds = 30
        self.time = Label(self.root, text=f'Time Left: {self.num_seconds}')
        self.time.grid(row=6, column=3, columnspan=2)
        # 居中显示
        self.root.withdraw()
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - self.root.winfo_reqwidth()) / 2
        y = (self.root.winfo_screenheight() - self.root.winfo_reqheight()) / 2
        self.root.geometry('+%d+%d' % (x, y))
        self.root.deiconify()
        # 计时
        self.tick()
    '''运行游戏'''
    def run(self):
        # 显示主界面
        self.root.mainloop()
    '''点击回调函数'''
    def clickcallback(self, event):
        card = event.widget
        if card.show: return
        # 之前没有卡片被翻开
        if len(self.shown_cards) == 0:
            self.shown_cards.append(card)
            image = ImageTk.PhotoImage(Image.open(os.path.join(self.card_dir, card.file+'.png')))
            card.configure(image=image)
            card.show_image = image
            card.show = True
        # 之前只有一张卡片被翻开
        elif len(self.shown_cards) == 1:
            # --之前翻开的卡片和现在的卡片一样
            if self.shown_cards[0].file == card.file:
                def delaycallback():
                    self.shown_cards[0].configure(image=self.blank_image)
                    self.shown_cards[0].blank_image = self.blank_image
                    card.configure(image=self.blank_image)
                    card.blank_image = self.blank_image
                    self.shown_cards.pop(0)
                    self.score_sound.play()
                self.num_existing_cards -= 2
                image = ImageTk.PhotoImage(Image.open(os.path.join(self.card_dir, card.file+'.png')))
                card.configure(image=image)
                card.show_image = image
                card.show = True
                card.after(300, delaycallback)
            # --之前翻开的卡片和现在的卡片不一样
            else:
                self.shown_cards.append(card)
                image = ImageTk.PhotoImage(Image.open(os.path.join(self.card_dir, card.file+'.png')))
                card.configure(image=image)
                card.show_image = image
                card.show = True
        # 之前有两张卡片被翻开
        elif len(self.shown_cards) == 2:
            # --之前翻开的第一张卡片和现在的卡片一样
            if self.shown_cards[0].file == card.file:
                def delaycallback():
                    self.shown_cards[0].configure(image=self.blank_image)
                    self.shown_cards[0].blank_image = self.blank_image
                    card.configure(image=self.blank_image)
                    card.blank_image = self.blank_image
                    self.shown_cards.pop(0)
                    self.score_sound.play()
                self.num_existing_cards -= 2
                image = ImageTk.PhotoImage(Image.open(os.path.join(self.card_dir, card.file+'.png')))
                card.configure(image=image)
                card.show_image = image
                card.show = True
                card.after(300, delaycallback)
            # --之前翻开的第二张卡片和现在的卡片一样
            elif self.shown_cards[1].file == card.file:
                def delaycallback():
                    self.shown_cards[1].configure(image=self.blank_image)
                    self.shown_cards[1].blank_image = self.blank_image
                    card.configure(image=self.blank_image)
                    card.blank_image = self.blank_image
                    self.shown_cards.pop(1)
                    self.score_sound.play()
                self.num_existing_cards -= 2
                image = ImageTk.PhotoImage(Image.open(os.path.join(self.card_dir, card.file+'.png')))
                card.configure(image=image)
                card.show_image = image
                card.show = True
                card.after(300, delaycallback)
            # --之前翻开的卡片和现在的卡片都不一样
            else:
                self.shown_cards.append(card)
                self.shown_cards[0].configure(image=self.cards_back_image)
                self.shown_cards[0].show = False
                self.shown_cards.pop(0)
                image = ImageTk.PhotoImage(Image.open(os.path.join(self.card_dir, card.file+'.png')))
                self.shown_cards[-1].configure(image=image)
                self.shown_cards[-1].show_image = image
                self.shown_cards[-1].show = True
        # 判断游戏是否已经胜利
        if self.num_existing_cards == 0:
            is_restart = messagebox.askyesno('Game Over', 'Congratulations, you win, do you want to play again?')
            if is_restart: self.restart()
            else: self.root.destroy()
    '''播放背景音乐'''
    def playbgm(self):
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(self.cfg.AUDIOPATHS['bgm'])
        pygame.mixer.music.play(-1, 0.0)
    '''计时'''
    def tick(self):
        if self.num_existing_cards == 0: return
        if self.num_seconds != 0:
            self.num_seconds -= 1
            self.time['text'] = f'Time Left: {self.num_seconds}'
            self.time.after(1000, self.tick)
        else:
            is_restart = messagebox.askyesno('Game Over', 'You fail since time up, do you want to play again?')
            if is_restart: self.restart()
            else: self.root.destroy()
    '''重新开始游戏'''
    def restart(self):
        self.root.destroy()
        client = FlipCardByMemoryGame()
        client.run()