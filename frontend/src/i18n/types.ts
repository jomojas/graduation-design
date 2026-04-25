export type AppLanguage = 'zh-CN' | 'en-US'

export type TranslationValue = string | ((...args: any[]) => string)

export type TranslationCatalog = Record<string, TranslationValue>
