import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class PromptService {
  constructor(private http: HttpClient) {}

  generatePrompts() {
    return this.http.get<any>('https://harry-ai-production.up.railway.app/prompt-styles');
  }
}
