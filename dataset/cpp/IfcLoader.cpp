/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */
 
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <format>
#include <fast_float/fast_float.h>
#include <spdlog/spdlog.h>
#include "IfcLoader.h"
#include "../version.h"
#include "../schema/IfcSchemaManager.h" 

namespace webifc::parsing {

  void p21encode(std::string_view input, std::ostringstream &output);
  std::string p21decode(std::string_view & str);    
 
   IfcLoader::IfcLoader(uint32_t tapeSize, uint32_t memoryLimit,schema::IfcSchemaManager &schemaManager) :_schemaManager(schemaManager)
   { 
     _tokenStream = new IfcTokenStream(tapeSize,memoryLimit/tapeSize);
     _nullLine = new IfcLine();
     _nullLine->ifcType=0;
     _nullLine->tapeOffset=0;
   }  
   
   const std::vector<uint32_t> IfcLoader::GetExpressIDsWithType(const uint32_t type) const
   { 
      if (_ifcTypeToExpressID.count(type)==0) return {};
      return _ifcTypeToExpressID.at(type);
   }
   
   const std::vector<uint32_t> IfcLoader::GetHeaderLinesWithType(const uint32_t type) const
   { 
     std::vector<uint32_t> ret;
     for (size_t i=0; i < _headerLines.size();i++)
     {
        if (_headerLines[i]->ifcType==type) ret.push_back(i);
     }
     return ret;
   }
   
   void IfcLoader::LoadFile(const std::function<uint32_t(char *, size_t, size_t)> &requestData)
   { 
     _tokenStream->SetTokenSource(requestData);
     ParseLines();
   }

   IFC_SCHEMA IfcLoader::GetSchema() const
   { 
      auto line = GetHeaderLinesWithType(schema::FILE_SCHEMA)[0];
      MoveToHeaderLineArgument(line, 0);
      auto schemas = _schemaManager.GetAvailableSchemas();

      while (!_tokenStream->IsAtEnd()) {
          IfcTokenType t = static_cast<IfcTokenType>(_tokenStream->Read<char>());
          if (t == IfcTokenType::LINE_END) break;
          if (t == IfcTokenType::LABEL) 
          {
            std::string_view schemaName = _tokenStream->ReadString();
            for (size_t i = 0; i < schemas.size();i++) 
            {
              if (_schemaManager.GetSchemaName(schemas[i]) == schemaName) return schemas[i];
            }
          }
      }
      return IFC2X3;
   }
   
   void IfcLoader::LoadFile(std::istream &requestData)
   { 
     _tokenStream->SetTokenSource(requestData);
     ParseLines();
   }
   
   void IfcLoader::SaveFile(const std::function<void(char *, size_t)> &outputData) const
   { 
      std::ostringstream output;
      output << "ISO-10303-21;"<<std::endl<<"HEADER;"<<std::endl;
      output << "/******************************************************" << std::endl;
      output << "* STEP Physical File produced by: IFCjs WebIfc " << WEB_IFC_VERSION_NUMBER << std::endl;
      output << "* Module: web-ifc/IfcLoader" << std::endl;
      output << "* Version: " << WEB_IFC_VERSION_NUMBER << std::endl;
      output << "* Source: https://github.com/IFCjs/web-ifc" << std::endl;
      output << "* Issues: https://github.com/IFCjs/web-ifc/issues" << std::endl;
      output << "******************************************************/" << std::endl;
      
      const std::vector<IfcLine*> *totalLines[2] = {&_headerLines, &_lines};
      for (uint8_t z=0; z < 2; z++)
      {
        const std::vector<IfcLine*>* currentLines = totalLines[z];
        for(uint32_t i=0; i < currentLines->size();i++)
        {
          IfcLine * line = (*currentLines)[i];

          if (line == _nullLine || line->ifcType == 0) continue;
          _tokenStream->MoveTo(line->tapeOffset);
          bool newLine = true;
          bool insideSet = false;
          IfcTokenType prev = IfcTokenType::EMPTY;
          while (!_tokenStream->IsAtEnd())
          {
            IfcTokenType t = static_cast<IfcTokenType>(_tokenStream->Read<char>());

            if (t != IfcTokenType::SET_END && t != IfcTokenType::LINE_END)
            {
              if (insideSet && prev != IfcTokenType::SET_BEGIN && prev != IfcTokenType::LABEL && prev != IfcTokenType::LINE_END)
              {
                output << ",";
              }
            }

            if (t == IfcTokenType::LINE_END)
            {
              output << ";" << std::endl;
              break;
            }

            switch (t)
            {
              case IfcTokenType::UNKNOWN:
              {
                output << "*";
                break;
              }
              case IfcTokenType::EMPTY:
              {
                output << "$";
                break;
              }
              case IfcTokenType::SET_BEGIN:
              {
                output << "(";
                insideSet = true;
                break;
              }
              case IfcTokenType::SET_END:
              {
                output << ")";
                break;
              }
              case IfcTokenType::STRING:
              {
                output << "'";
                p21encode(_tokenStream->ReadString(),output);
                output << "'";
                break;
              }
              case IfcTokenType::ENUM:
              {
                output << "." << _tokenStream->ReadString() << ".";
                break;
              }
              case IfcTokenType::REF:
              {
                output << "#" << _tokenStream->Read<uint32_t>();
                if (newLine) output << "=";
                break;
              }
              case IfcTokenType::LABEL:
              case IfcTokenType::REAL:
              case IfcTokenType::INTEGER:
              { 
                output << _tokenStream->ReadString();
                break;
              }
              default:
                break;
            }

            if (t == IfcTokenType::LINE_END)
            {
              newLine = true;
              insideSet = false;
            }
            else
            {
              newLine = false;
            }
            prev = t;
          }
        }
        if (z==0) output << "ENDSEC;"<<std::endl<<"DATA;"<<std::endl;
      }
      output << "ENDSEC;"<<std::endl<<"END-ISO-10303-21;";
      std::string tmp = output.str();
      outputData((char*)tmp.c_str(),tmp.size());
   }
   
   void IfcLoader::SaveFile(std::ostream &outputData) const
   { 
     SaveFile([&](char* src, size_t srcSize)
      {
          outputData.write(src,srcSize);
      }
    );
   }
      
   bool IfcLoader::IsAtEnd() const
   {
     return _tokenStream->IsAtEnd();
   }
  
   void IfcLoader::ParseLines() 
   {
        uint32_t maxExpressId = 0;
  			uint32_t currentIfcType = 0;
  			uint32_t currentExpressID = 0;
  			uint32_t currentTapeOffset = 0;
  			while (!_tokenStream->IsAtEnd())
  			{
          IfcTokenType t = static_cast<IfcTokenType>(_tokenStream->Read<char>());
  				switch (t)
  				{
  				case IfcTokenType::LINE_END:
  				{
            if (currentIfcType !=0)
  					{
  						IfcLine* l = new IfcLine();
  						l->ifcType = currentIfcType;
  						l->tapeOffset = currentTapeOffset;
  						if(currentIfcType == webifc::schema::FILE_DESCRIPTION || currentIfcType == webifc::schema::FILE_NAME || currentIfcType == webifc::schema::FILE_SCHEMA )
              {
                _headerLines.push_back(l);
              }
              else if (currentExpressID != 0)
              {
                _ifcTypeToExpressID[currentIfcType].push_back(currentExpressID);
                maxExpressId = std::max(maxExpressId, currentExpressID);
                _lines.resize(maxExpressId,_nullLine);
                _lines[currentExpressID-1]=l;
                currentExpressID = 0;
              }
              currentIfcType = 0;
  					}
  					currentTapeOffset = _tokenStream->GetReadOffset();
  					break;
  				}
  				case IfcTokenType::UNKNOWN:
  				case IfcTokenType::EMPTY:
  				case IfcTokenType::SET_BEGIN:
  				case IfcTokenType::SET_END:
  					break;
  				case IfcTokenType::STRING:
          case IfcTokenType::REAL:
          case IfcTokenType::INTEGER:
  				case IfcTokenType::ENUM:
  				{
  					auto size = _tokenStream->Read<uint16_t>();
            _tokenStream->Forward(size);
  					break;
  				}
  				case IfcTokenType::LABEL:
  				{
  					std::string_view s = _tokenStream->ReadString();
  					if (currentIfcType == 0) currentIfcType = _schemaManager.IfcTypeToTypeCode(s);
  					break;
  				}
  				case IfcTokenType::REF:
  				{
  					uint32_t ref = _tokenStream->Read<uint32_t>();
  					if (currentExpressID == 0) currentExpressID = ref;
  					break;
  				}
  				default:
  					break;
  				}
  			}
   }
   
   uint32_t IfcLoader::GetMaxExpressId() const
   { 
      return _lines.size();
   }
   
   bool IfcLoader::IsValidExpressID(const uint32_t expressID) const
   {  
   	 if (expressID == 0 || expressID > _lines.size() || _lines[expressID-1]==_nullLine) return false;
     else return true;
   }
   
   uint32_t IfcLoader::GetLineType(const uint32_t expressID) const
   { 
      if (expressID == 0 || expressID > _lines.size()) {
        spdlog::error("[GetLineType()] Attempt to Access Invalid ExpressID {}", expressID);
        return 0;
      }
      return _lines[expressID-1]->ifcType;
   }
   
   IfcLoader::~IfcLoader()
   { 
     delete _tokenStream;
   }
   
   void IfcLoader::MoveToLineArgument(const uint32_t expressID, const uint32_t argumentIndex) const
   { 
     _tokenStream->MoveTo(_lines[expressID-1]->tapeOffset);
     ArgumentOffset(argumentIndex);
   }
   
   void IfcLoader::MoveToHeaderLineArgument(const uint32_t lineID, const uint32_t argumentIndex) const
   { 
     _tokenStream->MoveTo(_headerLines[lineID]->tapeOffset);
   	 ArgumentOffset(argumentIndex);	
   }
   
   std::string_view IfcLoader::GetStringArgument() const
   { 
   	 _tokenStream->Read<char>(); // string type
     return _tokenStream->ReadString();
   }

   std::string IfcLoader::GetDecodedStringArgument() const
   { 
      std::string_view str = GetStringArgument();
      return p21decode(str);
   }

   void IfcLoader::PushDouble(double input)
   {             
      std::string numberString = std::format("{}", input);
      if (std::floor(input) == input) numberString+='.';
      size_t eLoc = numberString.find_first_of('e');
      if (eLoc != std::string::npos) numberString[eLoc]='E';
      uint16_t length = numberString.size();
      Push<uint16_t>((uint16_t)length);
      Push((void*)numberString.c_str(), numberString.size());        
   }

   void IfcLoader::PushInt(int input)
   {
    std::string numberString = std::to_string(input);
    uint16_t length = numberString.size();
    Push<uint16_t>((uint16_t)length);
    Push((void*)numberString.c_str(), numberString.size());             
   } 
   
   double IfcLoader::GetDoubleArgument() const
   { 
      std::string_view str = GetStringArgument();
      double number_value;
      fast_float::from_chars(str.data(), str.data() +str.size(), number_value);
      return number_value;
   }

   std::string_view  IfcLoader::GetDoubleArgumentAsString() const
   {
      return GetStringArgument();
   }

   int IfcLoader::GetIntArgument() const
   {
       std::string_view str = GetStringArgument();
       return std::stoi(std::string(str));
   }

  int IfcLoader::GetIntArgument(const uint32_t tapeOffset) const
  {
    _tokenStream->MoveTo(tapeOffset);
    return GetIntArgument();
  }

  uint32_t IfcLoader::GetCurrentLineExpressID() const
  {
      if (_lines.size()==0) return 0;
      uint32_t pos = _tokenStream->GetReadOffset();
      uint32_t prevLine = 0;
      for (size_t i=0; i < _lines.size();i++)
      {
        if (_lines[i] == _nullLine) continue;
        if (_lines[i]->tapeOffset > pos) break;
        prevLine = i;
      }
      return prevLine+1;
  }
   
   uint32_t IfcLoader::GetRefArgument() const
   { 
      if (_tokenStream->Read<char>() != IfcTokenType::REF)
     	{
     		spdlog::error("[GetRefArgument()] unexpected token type, expected REF {}", GetCurrentLineExpressID());
     		return 0;
     	}
     	return _tokenStream->Read<uint32_t>();
   }
   
  uint32_t IfcLoader::GetRefArgument(const uint32_t tapeOffset) const
	{
			_tokenStream->MoveTo(tapeOffset);
			return GetRefArgument();
	}
    
  double IfcLoader::GetDoubleArgument(const uint32_t tapeOffset) const
	{
		_tokenStream->MoveTo(tapeOffset);
		return GetDoubleArgument();
	}

  void IfcLoader::RemoveLine(const uint32_t expressID)
  {
      _lines[expressID-1]->ifcType = 0;
  }

  void IfcLoader::ExtendLineStorage(uint32_t lineStorageSize)
  {
    _lines.reserve(_lines.size()+lineStorageSize);
  }
  
  void IfcLoader::UpdateLineTape(const uint32_t expressID, const uint32_t type, const uint32_t start)
  {
    if (expressID > _lines.size())
  	{
      // create line object
  		IfcLine * line = new IfcLine();
      _lines.push_back(line);
  		// fill line data
  		line->ifcType = type;
      line->tapeOffset = start;
  		_ifcTypeToExpressID[type].push_back(expressID);

  	} else _lines[expressID-1]->tapeOffset = start;
  }

  void IfcLoader::AddHeaderLineTape(const uint32_t type, const uint32_t start)
  {
    
      IfcLine *l = new IfcLine();
      l->ifcType = type;
      l->tapeOffset = start;
      _headerLines.push_back(l);
  }
  
  IfcTokenType IfcLoader::GetTokenType(uint32_t tapeOffset) const
  {
    _tokenStream->MoveTo(tapeOffset);
    return GetTokenType();
  }
   
   uint32_t IfcLoader::GetOptionalRefArgument() const
   { 
      IfcTokenType t = GetTokenType();
     	if (t == IfcTokenType::EMPTY)
     	{
     		return 0;
     	}
     	else if (t == IfcTokenType::REF)
     	{
     		return _tokenStream->Read<uint32_t>();
     	}
     	else
     	{
     		spdlog::error("[GetOptionalRefArgument()] unexpected token type, expected REF or EMPTY", GetCurrentLineExpressID());
     		return 0;
     	}
   }
   
   IfcTokenType IfcLoader::GetTokenType() const
   { 
     return static_cast<IfcTokenType>(_tokenStream->Read<char>());
   }

   void IfcLoader::Push(void *v, uint64_t size)
   {
     _tokenStream->Push(v,size);
   }
   
   uint64_t IfcLoader::GetTotalSize()  const
   {
     return _tokenStream->GetTotalSize();
   }
     
   const std::vector<uint32_t> IfcLoader::GetSetArgument() const
   { 
     std::vector<uint32_t> tapeOffsets;
     _tokenStream->Read<char>(); // set begin
     int depth = 1;
     while (true)
     {
       uint32_t offset = _tokenStream->GetReadOffset();
       IfcTokenType t = static_cast<IfcTokenType>(_tokenStream->Read<char>());

       if (t == IfcTokenType::SET_BEGIN)
       {
         depth++;
       }
       else if (t == IfcTokenType::SET_END)
       {
         depth--;
       }
       else
       {
         tapeOffsets.push_back(offset);

         if (t == IfcTokenType::REF)
         {
           _tokenStream->Read<uint32_t>();
         }
         else if (t == IfcTokenType::STRING || t == IfcTokenType::INTEGER || t == IfcTokenType::REAL || t == IfcTokenType::LABEL || t == IfcTokenType::ENUM)
         {
           uint16_t length = _tokenStream->Read<uint16_t>();
           _tokenStream->Forward(length);
         }
         else
         {
           spdlog::error("[GetSetArgument[]) unexpected token", GetCurrentLineExpressID());
         }
       }

       if (depth == 0)
       {
         break;
       }
     }

     return tapeOffsets;
   }
   
   const std::vector<std::vector<uint32_t>> IfcLoader::GetSetListArgument() const
   { 
     std::vector<std::vector<uint32_t>> tapeOffsets;
   	 _tokenStream->Read<char>(); // set begin
   	 int depth = 1;
   	 std::vector<uint32_t> tempSet;

     	while (true)
     	{
     		uint32_t offset = _tokenStream->GetReadOffset();
     		IfcTokenType t = static_cast<IfcTokenType>(_tokenStream->Read<char>());

     		if (t == IfcTokenType::SET_BEGIN)
     		{
     			tempSet = std::vector<uint32_t>();
     			depth++;
     		}
     		else if (t == IfcTokenType::SET_END)
     		{
     			if (tempSet.size() > 0)
     			{
     				tapeOffsets.push_back(tempSet);
     				tempSet = std::vector<uint32_t>();
     			}
     			depth--;
     		}
     		else
     		{
     			tempSet.push_back(offset);

     			if (t == IfcTokenType::REF)
     			{
     				_tokenStream->Read<uint32_t>();
     			}
     			else if (t == IfcTokenType::STRING || t == IfcTokenType::INTEGER || t == IfcTokenType::REAL || t == IfcTokenType::LABEL || t == IfcTokenType::ENUM)
     			{
     				uint16_t length = _tokenStream->Read<uint16_t>();
     				_tokenStream->Forward(length);
     			}
     			else
     			{
     				spdlog::error("[GetSetListArgument()] unexpected token", GetCurrentLineExpressID());
     			}
     		}

     		if (depth == 0)
     		{
     			break;
     		}
     	}

     	return tapeOffsets;
   }
    
   void IfcLoader::ArgumentOffset(const uint32_t argumentIndex) const
   {
   	uint32_t movedOver = 0;
   	uint32_t setDepth = 0;
   	while (true)
   	{
   		if (setDepth == 1)
   		{
   			movedOver++;

   			if (movedOver-1 == argumentIndex)
   			{
   				return;
   			}
   		}

   		IfcTokenType t = static_cast<IfcTokenType>(_tokenStream->Read<char>());

   		switch (t)
   		{
   		case IfcTokenType::LINE_END:
   		{
   			spdlog::error("[ArgumentOffset()] unexpected line end {}", GetCurrentLineExpressID());
   			break;
   		}
   		case IfcTokenType::UNKNOWN:
   		case IfcTokenType::EMPTY:
   			break;
   		case IfcTokenType::SET_BEGIN:
   			setDepth++;
   			break;
   		case IfcTokenType::SET_END:
   			setDepth--;
   			if (setDepth == 0)
   			{
   				return;
   			}
   			break;
   		case IfcTokenType::STRING:
   		case IfcTokenType::ENUM:
   		case IfcTokenType::LABEL:
      case IfcTokenType::INTEGER:
      case IfcTokenType::REAL:
   		{
   			uint16_t length = _tokenStream->Read<uint16_t>();
   			_tokenStream->Forward(length);
   			break;
   		}
   		case IfcTokenType::REF:
   		{
   			_tokenStream->Read<uint32_t>();
   			break;
   		}
   		default:
   			break;
   		}
   	}
   }
   
   void IfcLoader::MoveToArgumentOffset(const uint32_t expressID, const uint32_t argumentIndex) const
   {
    _tokenStream->MoveTo(_lines[expressID-1]->tapeOffset);
   	ArgumentOffset(argumentIndex);
   }
   
   void IfcLoader::StepBack() const {
     _tokenStream->Back();
   }

    double IfcLoader::GetOptionalDoubleParam(double defaultValue = 0) const
    {
      if (GetTokenType() == IfcTokenType::REAL)
      {
        StepBack();
        return GetDoubleArgument();
      }
      StepBack();
      return defaultValue;
    }

}