import { enUS } from './en-US'
import { zhCN } from './zh-CN'
import type { AppLanguage, TranslationCatalog, TranslationValue } from './types'

export const LANGUAGE_STORAGE_KEY = 'ct-pet-language'

export const catalogs: Record<AppLanguage, TranslationCatalog> = {
  'zh-CN': zhCN,
  'en-US': enUS
}

export const detectBrowserLanguage = (): AppLanguage => {
  const language = navigator.language.toLowerCase()
  return language.startsWith('zh') ? 'zh-CN' : 'en-US'
}

export const isValidLanguage = (value: string | null): value is AppLanguage => {
  return value === 'zh-CN' || value === 'en-US'
}

export const translate = (
  catalog: TranslationCatalog,
  key: string,
  ...args: Array<string | number | null | undefined>
): string => {
  const value: TranslationValue | undefined = catalog[key]
  if (typeof value === 'function') {
    return value(...args)
  }
  if (typeof value === 'string') {
    return value
  }
  return key
}
