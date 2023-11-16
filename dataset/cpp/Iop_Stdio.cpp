#include <cstring>
#include "Iop_Stdio.h"
#include "Iop_Ioman.h"
#include "lexical_cast_ex.h"
#include "string_format.h"
#include "../Log.h"

#define LOG_NAME "iop_stdio"

#define FUNCTION_PRINTF "printf"
#define FUNCTION_PUTS "puts"

using namespace Iop;

CStdio::CStdio(uint8* ram, CIoman& ioman)
    : m_ram(ram)
    , m_ioman(ioman)
{
}

std::string CStdio::GetId() const
{
	return "stdio";
}

std::string CStdio::GetFunctionName(unsigned int functionId) const
{
	switch(functionId)
	{
	case 4:
		return FUNCTION_PRINTF;
		break;
	case 7:
		return FUNCTION_PUTS;
		break;
	default:
		return "unknown";
		break;
	}
}

void CStdio::Invoke(CMIPS& context, unsigned int functionId)
{
	switch(functionId)
	{
	case 4:
		__printf(context);
		break;
	case 7:
		context.m_State.nGPR[CMIPS::V0].nD0 = __puts(
		    context.m_State.nGPR[CMIPS::A0].nV0);
		break;
	default:
		CLog::GetInstance().Warn(LOG_NAME, "Unknown function (%d) called. PC = (%08X).\r\n",
		                         functionId, context.m_State.nPC);
		break;
	}
}

std::string CStdio::PrintFormatted(const char* format, CArgumentIterator& args)
{
	std::string output;
	while(*format != 0)
	{
		char character = *(format++);
		if(character == '%')
		{
			bool paramDone = false;
			bool inPrecision = false;
			bool showSign = false;
			char fillChar = ' ';
			std::string precision;
			while(!paramDone && *format != 0)
			{
				char type = *(format++);
				if(type == '%')
				{
					output += type;
					paramDone = true;
				}
				else if(type == '+')
				{
					showSign = true;
				}
				else if(type == 's')
				{
					const char* text = reinterpret_cast<const char*>(&m_ram[args.GetNext()]);
					output += text;
					paramDone = true;
				}
				else if(type == 'c')
				{
					char character = static_cast<char>(args.GetNext());
					output += character;
					paramDone = true;
				}
				else if(type == 'd' || type == 'i')
				{
					int number = args.GetNext();
					unsigned int precisionValue = precision.length() ? std::stoul(precision) : 1;
					if(showSign && (number >= 0))
					{
						output += "+";
					}
					output += lexical_cast_int<std::string>(number, precisionValue, fillChar);
					paramDone = true;
				}
				else if(type == 'u')
				{
					unsigned int number = args.GetNext();
					unsigned int precisionValue = precision.length() ? std::stoul(precision) : 1;
					output += lexical_cast_uint<std::string>(number, precisionValue);
					paramDone = true;
				}
				else if(type == 'x' || type == 'X' || type == 'p')
				{
					uint32 number = args.GetNext();
					std::string format;
					if(precision.empty())
					{
						format = string_format("%%%c", type);
					}
					else
					{
						unsigned int precisionValue = atoi(precision.c_str());
						format = string_format("%%0%d%c", precisionValue, type);
					}
					output += string_format(format.c_str(), number);
					paramDone = true;
				}
				else if(type == 'l')
				{
					//Length specifier, don't bother about it.
				}
				else if(type == '.')
				{
					inPrecision = true;
				}
				else
				{
					assert(isdigit(type));
					if(inPrecision)
					{
						precision += type;
					}
					else
					{
						fillChar = '0';
						inPrecision = true;
					}
				}
			}
		}
		else
		{
			output += character;
		}
	}
	return output;
}

void CStdio::__printf(CMIPS& context)
{
	CCallArgumentIterator args(context);
	auto format = reinterpret_cast<const char*>(m_ram + args.GetNext());
	auto output = PrintFormatted(format, args);
	m_ioman.Write(CIoman::FID_STDOUT, output.length(), output.c_str());
}

int32 CStdio::__puts(uint32 stringPtr)
{
	auto string = reinterpret_cast<const char*>(m_ram + stringPtr);
	m_ioman.Write(CIoman::FID_STDOUT, strlen(string), string);
	return 0;
}
