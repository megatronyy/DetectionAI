#ifndef LANG_H
#define LANG_H

#include <QString>

class Lang
{
public:
    enum Language { Chinese, English };
    static void setLanguage(Language lang);
    static Language language();
    static QString s(const QString& key);

private:
    static Language lang_;
};

#endif // LANG_H
