import "./global.css"
import { ZenMono } from "@hanzo/font/mono"
import { ZenSans } from "@hanzo/font/sans"
import { RootProvider } from "fumadocs-ui/provider/next"
import type { ReactNode } from "react"

export const metadata = {
  title: {
    default: "Lux EVM Documentation",
    template: "%s | Lux EVM",
  },
  description: "EVM implementation with Lux-specific optimizations",
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      className={`${ZenSans.variable} ${ZenMono.variable}`}
      suppressHydrationWarning
    >
      <body className="min-h-svh bg-background font-sans antialiased">
        <RootProvider
          search={{
            enabled: true,
          }}
          theme={{
            enabled: true,
            defaultTheme: "dark",
          }}
        >
          <div className="relative flex min-h-svh flex-col bg-background">
            {children}
          </div>
        </RootProvider>
      </body>
    </html>
  )
}
