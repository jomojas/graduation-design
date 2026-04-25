import React, { createContext, useContext, useMemo, useState } from 'react'
import { LANGUAGE_STORAGE_KEY, catalogs, detectBrowserLanguage, isValidLanguage, translate } from './index'
import type { AppLanguage } from './types'

type LanguageContextValue = {
  language: AppLanguage
  setLanguage: (language: AppLanguage) => void
  t: (key: string, ...args: Array<string | number | null | undefined>) => string
}

const LanguageContext = createContext<LanguageContextValue | null>(null)

const getInitialLanguage = (): AppLanguage => {
  const saved = localStorage.getItem(LANGUAGE_STORAGE_KEY)
  if (isValidLanguage(saved)) {
    return saved
  }
  return detectBrowserLanguage()
}

export const LanguageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [language, setLanguageState] = useState<AppLanguage>(() => getInitialLanguage())

  const setLanguage = (next: AppLanguage) => {
    setLanguageState(next)
    localStorage.setItem(LANGUAGE_STORAGE_KEY, next)
  }

  const value = useMemo<LanguageContextValue>(() => {
    const catalog = catalogs[language]
    return {
      language,
      setLanguage,
      t: (key, ...args) => translate(catalog, key, ...args)
    }
  }, [language])

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>
}

export const useLanguage = () => {
  const context = useContext(LanguageContext)
  if (!context) {
    throw new Error('useLanguage must be used within LanguageProvider')
  }
  return context
}
