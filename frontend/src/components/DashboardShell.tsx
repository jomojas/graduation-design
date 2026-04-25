import React from 'react'
import { useLanguage } from '../i18n/LanguageProvider'

export function DashboardShell({ children }: { children?: React.ReactNode }) {
  const { language, setLanguage, t } = useLanguage()

  return (
    <div className="medical-console-layout">
      <header className="medical-console-header">
        <div className="medical-console-header-top">
          <span className="medical-console-chip">{t('shellWorkspaceChip')}</span>
          <span className="medical-console-chip medical-console-chip-muted">{t('shellNiivueChip')}</span>
          <div className="medical-language-switch" role="group" aria-label="language-switcher">
            <button
              type="button"
              className={`medical-language-button ${language === 'zh-CN' ? 'medical-language-button-active' : ''}`}
              onClick={() => setLanguage('zh-CN')}
            >
              {t('languageChinese')}
            </button>
            <button
              type="button"
              className={`medical-language-button ${language === 'en-US' ? 'medical-language-button-active' : ''}`}
              onClick={() => setLanguage('en-US')}
            >
              {t('languageEnglish')}
            </button>
          </div>
        </div>
        <h1 className="medical-console-title">{t('shellTitle')}</h1>
        <p className="medical-console-subtitle">{t('shellSubtitle')}</p>
        <p className="medical-console-caption">{t('shellCaption')}</p>
      </header>
      <main>{children}</main>
    </div>
  )
}
