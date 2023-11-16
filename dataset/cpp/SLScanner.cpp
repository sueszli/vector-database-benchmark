/*
 * SLScanner.cpp
 * 
 * This file is part of the XShaderCompiler project (Copyright (c) 2014-2018 by Lukas Hermanns)
 * See "LICENSE.txt" for license information.
 */

#include "SLScanner.h"
#include <cctype>


namespace Xsc
{


SLScanner::SLScanner(Log* log) :
    Scanner { log }
{
}

TokenPtr SLScanner::Next()
{
    return NextToken(false, false);
}


/*
 * ======= Private: =======
 */

TokenPtr SLScanner::ScanToken()
{
    std::string spell;

    /* Scan directive (beginning with '#') */
    if (Is('#'))
        return ScanDirective();

    /* Scan identifier */
    if (std::isalpha(UChr()) || Is('_'))
        return ScanIdentifier();

    /* Scan number */
    if (Is('.'))
        return ScanNumberOrDot();
    if (std::isdigit(UChr()))
        return ScanNumber();

    /* Scan string literal */
    if (Is('\"'))
        return ScanStringLiteral();

    /* Scan operators */
    if (Is('='))
    {
        spell += TakeIt();
        if (Is('='))
            return Make(Tokens::BinaryOp, spell, true);
        return Make(Tokens::AssignOp, spell);
    }

    if (Is('~'))
        return Make(Tokens::UnaryOp, spell, true);

    if (Is('!'))
    {
        spell += TakeIt();
        if (Is('='))
            return Make(Tokens::BinaryOp, spell, true);
        return Make(Tokens::UnaryOp, spell);
    }

    if (Is('%'))
    {
        spell += TakeIt();
        if (Is('='))
            return Make(Tokens::AssignOp, spell, true);
        return Make(Tokens::BinaryOp, spell);
    }

    if (Is('*'))
    {
        spell += TakeIt();
        if (Is('='))
            return Make(Tokens::AssignOp, spell, true);
        return Make(Tokens::BinaryOp, spell);
    }

    if (Is('^'))
    {
        spell += TakeIt();
        if (Is('='))
            return Make(Tokens::AssignOp, spell, true);
        return Make(Tokens::BinaryOp, spell);
    }

    if (Is('+'))
        return ScanPlusOp();
    if (Is('-'))
        return ScanMinusOp();

    if (Is('<') || Is('>'))
        return ScanAssignShiftRelationOp(Chr());

    if (Is('&'))
    {
        spell += TakeIt();
        if (Is('='))
            return Make(Tokens::AssignOp, spell, true);
        if (Is('&'))
            return Make(Tokens::BinaryOp, spell, true);
        return Make(Tokens::BinaryOp, spell);
    }

    if (Is('|'))
    {
        spell += TakeIt();
        if (Is('='))
            return Make(Tokens::AssignOp, spell, true);
        if (Is('|'))
            return Make(Tokens::BinaryOp, spell, true);
        return Make(Tokens::BinaryOp, spell);
    }

    if (Is(':'))
    {
        spell += TakeIt();
        if (Is(':'))
            return Make(Tokens::DColon, spell, true);
        return Make(Tokens::Colon, spell);
    }

    /* Scan punctuation, special characters and brackets */
    switch (Chr())
    {
        case ';': return Make(Tokens::Semicolon, true); break;
        case ',': return Make(Tokens::Comma,     true); break;
        case '?': return Make(Tokens::TernaryOp, true); break;
        case '(': return Make(Tokens::LBracket,  true); break;
        case ')': return Make(Tokens::RBracket,  true); break;
        case '{': return Make(Tokens::LCurly,    true); break;
        case '}': return Make(Tokens::RCurly,    true); break;
        case '[': return Make(Tokens::LParen,    true); break;
        case ']': return Make(Tokens::RParen,    true); break;
    }

    ErrorUnexpected();

    return nullptr;
}

TokenPtr SLScanner::ScanDirective()
{
    std::string spell;

    /* Take directive begin '#' */
    Take('#');

    /* Ignore white spaces (but not new-lines) */
    IgnoreWhiteSpaces(false);

    /* Scan identifier string */
    StoreStartPos();

    while (std::isalpha(UChr()))
        spell += TakeIt();

    /* Return as identifier */
    return Make(Token::Types::Directive, spell);
}

TokenPtr SLScanner::ScanIdentifier()
{
    /* Scan identifier string */
    std::string spell;
    spell += TakeIt();

    while (std::isalnum(UChr()) || Is('_'))
        spell += TakeIt();

    /* Scan identifier or keyword */
    return ScanIdentifierOrKeyword(std::move(spell));
}

TokenPtr SLScanner::ScanAssignShiftRelationOp(const char chr)
{
    std::string spell;
    spell += TakeIt();

    if (Is(chr))
    {
        spell += TakeIt();

        if (Is('='))
            return Make(Tokens::AssignOp, spell, true);

        return Make(Tokens::BinaryOp, spell);
    }

    if (Is('='))
        spell += TakeIt();

    return Make(Tokens::BinaryOp, spell);
}

TokenPtr SLScanner::ScanPlusOp()
{
    std::string spell;
    spell += TakeIt();

    if (Is('+'))
        return Make(Tokens::UnaryOp, spell, true);
    else if (Is('='))
        return Make(Tokens::AssignOp, spell, true);

    return Make(Tokens::BinaryOp, spell);
}

TokenPtr SLScanner::ScanMinusOp()
{
    std::string spell;
    spell += TakeIt();

    if (Is('-'))
        return Make(Tokens::UnaryOp, spell, true);
    else if (Is('='))
        return Make(Tokens::AssignOp, spell, true);

    return Make(Tokens::BinaryOp, spell);
}


} // /namespace Xsc



// ================================================================================