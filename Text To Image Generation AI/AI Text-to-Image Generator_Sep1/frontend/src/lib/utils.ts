import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

// Standard shadcn/ui helper: merge conditional class names, then let
// tailwind-merge resolve any conflicting Tailwind utility classes.
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
