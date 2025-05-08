import { Routes } from '@angular/router';

export const routes: Routes = [
{
    path: '',
    title: 'Chat Application',
    loadComponent: () => import('../components/chat-application/chat-application.component').then(m => m.ChatApplicationComponent)
  }
];
