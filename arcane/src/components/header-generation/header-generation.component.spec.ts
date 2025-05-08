import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HeaderGenerationComponent } from './header-generation.component';

describe('HeaderGenerationComponent', () => {
  let component: HeaderGenerationComponent;
  let fixture: ComponentFixture<HeaderGenerationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [HeaderGenerationComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(HeaderGenerationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
